#!/usr/bin/env python3
"""Collect an arXiv corpus of Huawei-domain papers via review-seed expansion.

Covers Huawei's technology stack per 2025 annual report and public research org
disclosures (2012 Labs / Noah's Ark Lab / Central Research Institute):

  Track 1 — Wireless & Networks (5G-A/6G, Massive MIMO, beamforming, AI-for-networks)
  Track 2 — Optical Communications (silicon photonics, optical networks, FEC)
  Track 3 — AI / ML Foundations (LLM, NLP, CV, recsys, RL, trustworthy AI, multimodal)
  Track 4 — Computing Infrastructure (distributed systems, AI infra, compilers, DB, cloud-native)
  Track 5 — Terminal / Consumer (on-device AI, mobile, computational photography, HCI)
  Track 6 — Digital Power (power electronics, energy storage, grid-forming, efficiency)
  Track 7 — Intelligent Automotive (autonomous driving, world models, planning, V2X)
  Track 8 — Chip / Semiconductor (architecture, interconnect, EDA)
  Track 9 — Materials & Devices (novel materials, quantum materials, nanoscale)

Pipeline (resume-friendly, each phase can run independently):
  P1 search_surveys      arXiv API category+keyword search for review/survey seeds
  P2 download_seeds      download PDF + LaTeX for the survey seeds
  P3 expand_refs         extract arXiv ref IDs from seeds' LaTeX (.bbl/.bib/.tex)
  P4 filter_domain       query arXiv metadata in batches; keep only papers whose
                         primary category is in the Huawei-domain whitelist
  P5 download_refs       download PDF + LaTeX for the filtered candidates,
                         until --target is reached or candidates exhausted

Usage:
    # Smoke run (~10 surveys, no ref expansion):
    python scripts/collect_huawei_domain_papers.py --phase smoke

    # Full pipeline (~3000 papers):
    python scripts/collect_huawei_domain_papers.py --phase all --target 3000

    # Smaller run:
    python scripts/collect_huawei_domain_papers.py --phase all --target 1000 --max-seeds 50

    # Per-phase resume:
    python scripts/collect_huawei_domain_papers.py --phase search_surveys
    python scripts/collect_huawei_domain_papers.py --phase download_seeds
    python scripts/collect_huawei_domain_papers.py --phase expand_refs
    python scripts/collect_huawei_domain_papers.py --phase filter_domain
    python scripts/collect_huawei_domain_papers.py --phase download_refs --target 3000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from batch_download_references import (  # noqa: E402
    download_latex,
    download_pdf,
    extract_arxiv_ids_from_latex,
)
from download_latex_sources import process_papers as download_seed_sources  # noqa: E402

# ── arXiv API ──────────────────────────────────────────────────────────
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ── Output paths ───────────────────────────────────────────────────────
DEFAULT_BASE = PROJECT_ROOT / "data" / "00_raw"
DEFAULT_SEED_SOURCE_DIR = DEFAULT_BASE / "huawei_seed_latex_sources"
DEFAULT_PDF_DIR = DEFAULT_BASE / "huawei_pdfs"
DEFAULT_LATEX_DIR = DEFAULT_BASE / "huawei_latex_sources"
DEFAULT_SEED_MANIFEST = DEFAULT_BASE / "huawei_seed_manifest.json"
DEFAULT_TARGET_IDS = DEFAULT_BASE / "huawei_arxiv_ids.txt"
DEFAULT_SUCCESS_IDS = DEFAULT_BASE / "huawei_success_ids.txt"
DEFAULT_MANIFEST = DEFAULT_BASE / "huawei_manifest.json"

REVIEW_KEYWORDS = ("review", "survey", "overview", "tutorial", "primer")

# ── Huawei domain: primary-category whitelist ──────────────────────────
# Papers whose *primary* arXiv category matches any of these prefixes
# are kept during the metadata-filter step (P4).  This list is the union
# of the categories mentioned in the per-track search queries below.
HUAWEI_CATEGORY_RE = re.compile(
    r"^("
    # Track 1 — Wireless & Networks
    r"cs\.IT|cs\.NI|eess\.SP|"
    # Track 2 — Optical Communications
    r"physics\.optics|"
    # Track 3 — AI / ML Foundations
    r"cs\.LG|cs\.AI|cs\.CL|cs\.CV|cs\.IR|cs\.NE|"
    # Track 4 — Computing Infrastructure
    r"cs\.DC|cs\.DB|cs\.OS|cs\.PL|cs\.SE|cs\.PF|"
    # Track 5 — Terminal / Consumer
    r"cs\.HC|cs\.MM|"
    # Track 6 — Digital Power
    r"eess\.SY|cs\.SY|math\.OC|"
    # Track 7 — Intelligent Automotive
    r"cs\.RO|cs\.MA|"
    # Track 8 — Chip / Semiconductor
    r"cs\.AR|cs\.ET|physics\.app-ph|"
    # Track 9 — Materials & Devices
    r"cond-mat\.mes-hall|cond-mat\.mtrl-sci"
    r")"
)

# ── Huawei-domain topic queries ────────────────────────────────────────
# Each tuple is (label, arXiv query string).
# Query format: all:review AND all:<keyword> AND cat:<category>
# We search for review/survey papers in each domain as seed points.

HUAWEI_TOPIC_QUERIES: List[tuple[str, str]] = [
    # ══════════════════════════════════════════════════════════════════
    # Track 1 — Wireless & Networks (运营商网络、无线接入、核心网)
    # ══════════════════════════════════════════════════════════════════
    ("wireless_mimo", "all:review AND all:massive AND all:MIMO AND cat:cs.IT"),
    ("wireless_channel", "all:review AND all:channel AND all:estimation AND cat:cs.IT"),
    ("wireless_beamforming", "all:review AND all:beamforming AND cat:eess.SP"),
    ("wireless_6g", "all:survey AND all:6G AND cat:cs.NI"),
    ("wireless_ai_networks", "all:review AND all:AI AND all:network AND (cat:cs.NI OR cat:cs.IT)"),
    ("wireless_optimization", "all:review AND all:resource AND all:allocation AND cat:cs.NI"),
    ("network_protocol", "all:survey AND all:network AND all:protocol AND cat:cs.NI"),
    ("network_deterministic", "all:review AND all:deterministic AND all:network AND cat:cs.NI"),

    # ══════════════════════════════════════════════════════════════════
    # Track 2 — Optical Communications (光通信、硅光子)
    # ══════════════════════════════════════════════════════════════════
    ("optical_communications", "all:review AND all:optical AND all:communication AND cat:physics.optics"),
    ("silicon_photonics", "all:review AND all:silicon AND all:photonics AND (cat:physics.optics OR cat:physics.app-ph)"),
    ("optical_network", "all:survey AND all:optical AND all:network AND cat:cs.NI"),
    ("optical_interconnect", "all:review AND all:optical AND all:interconnect AND (cat:physics.optics OR cat:cs.AR)"),

    # ══════════════════════════════════════════════════════════════════
    # Track 3 — AI / ML Foundations (诺亚方舟核心方向)
    # ══════════════════════════════════════════════════════════════════
    ("llm_pretraining", "all:survey AND all:large AND all:language AND all:model AND cat:cs.CL"),
    ("llm_efficient", "all:survey AND all:efficient AND all:language AND all:model AND (cat:cs.CL OR cat:cs.LG)"),
    ("multimodal_learning", "all:survey AND all:multimodal AND all:learning AND (cat:cs.CV OR cat:cs.CL OR cat:cs.LG)"),
    ("recommender_systems", "all:survey AND all:recommender AND all:system AND cat:cs.IR"),
    ("reinforcement_learning", "all:survey AND all:reinforcement AND all:learning AND cat:cs.LG"),
    ("trustworthy_ai", "all:survey AND all:trustworthy AND all:AI AND (cat:cs.LG OR cat:cs.AI)"),
    ("nlp_foundation", "all:survey AND all:natural AND all:language AND all:processing AND cat:cs.CL"),
    ("cv_foundation", "all:survey AND all:computer AND all:vision AND (cat:cs.CV OR cat:cs.AI)"),
    ("bayesian_optimization", "all:review AND all:Bayesian AND all:optimization AND (cat:cs.LG OR cat:stat.ML)"),
    ("knowledge_graph", "all:survey AND all:knowledge AND all:graph AND (cat:cs.AI OR cat:cs.CL OR cat:cs.IR)"),

    # ══════════════════════════════════════════════════════════════════
    # Track 4 — Computing Infrastructure (计算基础设施)
    # ══════════════════════════════════════════════════════════════════
    ("distributed_systems", "all:survey AND all:distributed AND all:system AND (cat:cs.DC OR cat:cs.OS)"),
    ("ai_infrastructure", "all:survey AND all:AI AND all:infrastructure AND (cat:cs.DC OR cat:cs.LG)"),
    ("compiler_optimization", "all:survey AND all:compiler AND all:optimization AND cat:cs.PL"),
    ("inference_serving", "all:survey AND all:LLM AND all:inference AND (cat:cs.LG OR cat:cs.DC)"),
    ("database_systems", "all:survey AND all:database AND all:system AND cat:cs.DB"),
    ("cloud_native", "all:survey AND all:cloud AND all:native AND (cat:cs.DC OR cat:cs.SE)"),
    ("model_compression", "all:survey AND all:model AND all:compression AND cat:cs.LG"),

    # ══════════════════════════════════════════════════════════════════
    # Track 5 — Terminal / Consumer (终端、移动影像、OS)
    # ══════════════════════════════════════════════════════════════════
    ("on_device_ai", "all:survey AND all:on-device AND all:AI AND (cat:cs.LG OR cat:cs.CV)"),
    ("edge_intelligence", "all:survey AND all:edge AND all:intelligence AND (cat:cs.NI OR cat:cs.DC)"),
    ("mobile_computing", "all:survey AND all:mobile AND all:computing AND (cat:cs.HC OR cat:cs.OS)"),
    ("computational_photography", "all:survey AND all:computational AND all:photography AND cat:cs.CV"),
    ("hci_multimodal", "all:survey AND all:multimodal AND all:interaction AND cat:cs.HC"),

    # ══════════════════════════════════════════════════════════════════
    # Track 6 — Digital Power (数字能源)
    # ══════════════════════════════════════════════════════════════════
    ("power_electronics", "all:review AND all:power AND all:electronics AND cat:eess.SY"),
    ("energy_storage", "all:review AND all:energy AND all:storage AND cat:eess.SY"),
    ("grid_forming", "all:review AND all:grid-forming AND (cat:eess.SY OR cat:cs.SY)"),
    ("energy_efficiency", "all:review AND all:energy AND all:efficiency AND (cat:eess.SY OR cat:cs.SY)"),
    ("smart_grid", "all:review AND all:smart AND all:grid AND (cat:eess.SY OR cat:math.OC)"),

    # ══════════════════════════════════════════════════════════════════
    # Track 7 — Intelligent Automotive (智能汽车)
    # ══════════════════════════════════════════════════════════════════
    ("autonomous_driving", "all:survey AND all:autonomous AND all:driving AND (cat:cs.RO OR cat:cs.CV)"),
    ("trajectory_prediction", "all:survey AND all:trajectory AND all:prediction AND (cat:cs.RO OR cat:cs.CV)"),
    ("motion_planning", "all:survey AND all:motion AND all:planning AND (cat:cs.RO OR cat:cs.AI)"),
    ("world_model", "all:survey AND all:world AND all:model AND all:autonomous AND (cat:cs.RO OR cat:cs.LG)"),
    ("v2x_communication", "all:survey AND all:V2X AND (cat:cs.NI OR cat:cs.RO)"),
    ("simulation_ad", "all:survey AND all:simulation AND all:autonomous AND (cat:cs.RO OR cat:cs.AI)"),

    # ══════════════════════════════════════════════════════════════════
    # Track 8 — Chip / Semiconductor (芯片、半导体)
    # ══════════════════════════════════════════════════════════════════
    ("chip_architecture", "all:survey AND all:chip AND all:architecture AND (cat:cs.AR OR cat:cs.ET)"),
    ("interconnect", "all:survey AND all:interconnect AND (cat:cs.AR OR cat:cond-mat.mes-hall)"),
    ("eda_design", "all:survey AND all:electronic AND all:design AND all:automation AND cat:cs.AR"),
    ("semiconductor_device", "all:review AND all:semiconductor AND all:device AND (cat:physics.app-ph OR cat:cond-mat.mes-hall)"),

    # ══════════════════════════════════════════════════════════════════
    # Track 9 — Materials & Devices (新材料、器件物理)
    # ══════════════════════════════════════════════════════════════════
    ("novel_materials", "all:review AND all:novel AND all:materials AND (cat:cond-mat.mtrl-sci OR cat:physics.app-ph)"),
    ("quantum_materials", "all:review AND all:quantum AND all:materials AND (cat:cond-mat.mtrl-sci OR cat:cond-mat.mes-hall)"),
    ("nanomaterials", "all:review AND all:nanomaterial AND cat:cond-mat.mtrl-sci"),
    ("2d_materials", "all:review AND all:2D AND all:materials AND (cat:cond-mat.mtrl-sci OR cat:cond-mat.mes-hall)"),

    # ══════════════════════════════════════════════════════════════════
    # Extended queries — deeper coverage for each track
    # ══════════════════════════════════════════════════════════════════

    # Track 1 extended
    ("wireless_coding", "all:survey AND all:channel AND all:coding AND cat:cs.IT"),
    ("wireless_noma", "all:survey AND all:NOMA AND cat:cs.IT"),
    ("wireless_ris", "all:survey AND all:reconfigurable AND all:intelligent AND all:surface AND cat:cs.IT"),

    # Track 2 extended
    ("fiber_optic", "all:review AND all:fiber AND all:optic AND cat:physics.optics"),
    ("photonic_integration", "all:review AND all:photonic AND all:integrated AND (cat:physics.optics OR cat:physics.app-ph)"),

    # Track 3 extended
    ("contrastive_learning", "all:survey AND all:contrastive AND all:learning AND cat:cs.LG"),
    ("graph_neural_networks", "all:survey AND all:graph AND all:neural AND all:network AND (cat:cs.LG OR cat:cs.AI)"),
    ("federated_learning", "all:survey AND all:federated AND all:learning AND (cat:cs.LG OR cat:cs.DC)"),
    ("information_retrieval", "all:survey AND all:dense AND all:retrieval AND cat:cs.IR"),
    ("speech_processing", "all:survey AND all:speech AND all:recognition AND (cat:cs.CL OR cat:eess.AS)"),

    # Track 4 extended
    ("gpu_programming", "all:survey AND all:GPU AND all:computing AND (cat:cs.DC OR cat:cs.AR)"),
    ("kv_cache", "all:survey AND all:KV AND all:cache AND cat:cs.LG"),
    ("vector_database", "all:survey AND all:vector AND all:database AND cat:cs.DB"),
    ("serverless", "all:survey AND all:serverless AND all:computing AND cat:cs.DC"),
    ("train_parallelism", "all:survey AND all:parallel AND all:training AND (cat:cs.LG OR cat:cs.DC)"),

    # Track 5 extended
    ("tinyml", "all:survey AND all:TinyML AND (cat:cs.LG OR cat:cs.AR)"),
    ("image_enhance", "all:survey AND all:image AND all:enhancement AND (cat:cs.CV OR cat:eess.IV)"),
    ("ar_vr", "all:survey AND all:augmented AND all:reality AND (cat:cs.HC OR cat:cs.CV)"),

    # Track 6 extended
    ("solar_photovoltaic", "all:review AND all:photovoltaic AND cat:physics.app-ph"),
    ("battery_management", "all:review AND all:battery AND all:management AND (cat:eess.SY OR cat:cs.SY)"),
    ("dc_microgrid", "all:review AND all:microgrid AND (cat:eess.SY OR cat:cs.SY)"),

    # Track 7 extended
    ("sensor_fusion", "all:survey AND all:sensor AND all:fusion AND (cat:cs.RO OR cat:cs.CV)"),
    ("end_to_end_ad", "all:survey AND all:end-to-end AND all:autonomous AND all:driving AND cat:cs.RO"),
    ("occupancy_network", "all:survey AND all:occupancy AND all:network AND cat:cs.CV"),

    # Track 8 extended
    ("neuromorphic", "all:review AND all:neuromorphic AND all:computing AND (cat:cs.ET OR cat:cs.AR)"),
    ("memory_architecture", "all:survey AND all:memory AND all:architecture AND cat:cs.AR"),
    ("quantum_computing", "all:review AND all:quantum AND all:computing AND (cat:quant-ph OR cat:cs.ET)"),

    # Track 9 extended
    ("metamaterials", "all:review AND all:metamaterial AND (cat:cond-mat.mtrl-sci OR cat:physics.app-ph)"),
    ("perovskite", "all:review AND all:perovskite AND (cat:cond-mat.mtrl-sci OR cat:physics.app-ph)"),
    ("topological_insulator", "all:review AND all:topological AND all:insulator AND cond-mat.mes-hall"),
]


# ── Helpers ────────────────────────────────────────────────────────────

def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def is_huawei_domain(categories: Sequence[str]) -> bool:
    """True if the paper's primary category is in the Huawei-domain whitelist."""
    if not categories:
        return False
    primary = categories[0].strip()
    return bool(HUAWEI_CATEGORY_RE.match(primary))


def parse_arxiv_entries(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        id_text = entry.findtext("atom:id", "", ARXIV_NS)
        if "/abs/" not in id_text:
            continue
        arxiv_id = strip_version(id_text.rsplit("/abs/", 1)[-1])
        title = " ".join((entry.findtext("atom:title", "", ARXIV_NS) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", "", ARXIV_NS) or "").split())
        published = (entry.findtext("atom:published", "", ARXIV_NS) or "")[:10]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ARXIV_NS)]
        authors = [
            a.findtext("atom:name", "", ARXIV_NS)
            for a in entry.findall("atom:author", ARXIV_NS)
        ]
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "published": published,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "categories": categories,
                "authors": [a for a in authors if a],
            }
        )
    return rows


def request_arxiv(
    session: requests.Session,
    params: Dict[str, Any],
    delay: float,
    max_retries: int,
) -> Optional[str]:
    for attempt in range(max_retries):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = session.get(ARXIV_API, params=params, timeout=60)
        except requests.RequestException as exc:
            wait = min(300, 20 * (attempt + 1))
            print(f"  [arxiv-api] network error: {exc}; wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 429:
            wait = min(600, 60 * (attempt + 1))
            print(f"  [arxiv-api] 429 rate limit; wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        print(f"  [arxiv-api] HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
        if resp.status_code >= 500:
            time.sleep(min(300, 20 * (attempt + 1)))
            continue
        return None
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{aid}\n" for aid in ids), encoding="utf-8")


def load_seed_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.seed_ids:
        ids.extend(args.seed_ids)
    if args.seed_id_file:
        for line in args.seed_id_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return [strip_version(aid) for aid in ids]


# ── P1: Search review seeds ────────────────────────────────────────────

def score_seed(row: Dict[str, Any]) -> float:
    title = (row.get("title") or "").lower()
    abstract = (row.get("abstract") or "").lower()
    score = 0.0
    if any(k in title for k in REVIEW_KEYWORDS):
        score += 10.0
    if "review" in title:
        score += 5.0
    if "survey" in title:
        score += 4.0
    if any(phrase in abstract for phrase in ("we review", "we survey", "this review", "this survey")):
        score += 3.0
    year = row.get("year") or 0
    if year:
        score += min(3.0, max(0.0, (year - 2015) / 4.0))
    return score


def search_review_seeds(args: argparse.Namespace) -> List[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-huawei-seed-search/1.0 (research)"})
    seen: Set[str] = set()
    candidates: List[Dict[str, Any]] = []

    for label, query in HUAWEI_TOPIC_QUERIES:
        print(f"[seed-search] {label}: {query}", flush=True)
        topic_added = 0
        xml_text = request_arxiv(
            session,
            {
                "search_query": query,
                "start": 0,
                "max_results": args.search_results_per_topic,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            delay=args.api_delay,
            max_retries=args.api_retries,
        )
        if not xml_text:
            print("  no response", flush=True)
            continue
        rows = parse_arxiv_entries(xml_text)
        print(f"  got {len(rows)} rows", flush=True)
        for row in rows:
            aid = row["arxiv_id"]
            if aid in seen:
                continue
            if not is_huawei_domain(row.get("categories", [])):
                continue
            hay = f"{row.get('title','')} {row.get('abstract','')}".lower()
            if not any(k in hay for k in REVIEW_KEYWORDS):
                continue
            row["seed_topic"] = label
            row["_review_score"] = score_seed(row)
            seen.add(aid)
            candidates.append(row)
            topic_added += 1
            if len(candidates) >= args.max_seeds:
                candidates.sort(key=lambda r: (r.get("_review_score", 0), r.get("year") or 0), reverse=True)
                return candidates
            if args.max_seeds_per_topic > 0 and topic_added >= args.max_seeds_per_topic:
                break

    candidates.sort(key=lambda r: (r.get("_review_score", 0), r.get("year") or 0), reverse=True)
    return candidates


# ── P2: Download seed sources ──────────────────────────────────────────

# (handled inline via download_seed_sources in collect_refs_from_seed_sources)


# ── P3: Expand references ──────────────────────────────────────────────

def collect_refs_from_seed_sources(
    seed_ids: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Set[str]]:
    print(f"[seed-source] downloading/extracting {len(seed_ids)} seed review sources", flush=True)
    download_seed_sources(
        arxiv_ids=list(seed_ids),
        output_dir=args.seed_source_dir,
        delay=args.download_delay,
        extract_only=False,
        verify_ssl=not args.no_verify,
    )
    extracted = args.seed_source_dir / "extracted"
    seed_refs = extract_arxiv_ids_from_latex(extracted)
    seed_set = set(seed_ids)
    return {seed: refs for seed, refs in seed_refs.items() if seed in seed_set}


# ── P4: Filter candidates to Huawei domain ─────────────────────────────

def fetch_metadata_for_ids(
    ids: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-huawei-metadata/1.0 (research)"})
    metadata: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50
    for start in range(0, len(ids), chunk_size):
        chunk = list(ids[start:start + chunk_size])
        print(f"[metadata] {start + 1}-{start + len(chunk)} / {len(ids)}", flush=True)
        xml_text = request_arxiv(
            session,
            {"id_list": ",".join(chunk), "max_results": len(chunk)},
            delay=args.api_delay,
            max_retries=args.api_retries,
        )
        if not xml_text:
            continue
        for row in parse_arxiv_entries(xml_text):
            metadata[row["arxiv_id"]] = row
    return metadata


def choose_target_candidates(
    seed_refs: Dict[str, Set[str]],
    seed_rows: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    all_ref_ids: List[str] = []
    seen: Set[str] = set(seed_refs.keys())
    for refs in seed_refs.values():
        for aid in sorted(strip_version(r) for r in refs):
            if aid not in seen:
                seen.add(aid)
                all_ref_ids.append(aid)

    print(f"[refs] unique refs before metadata/domain filter: {len(all_ref_ids)}", flush=True)
    if args.allow_unverified_refs:
        print("[refs] --allow-unverified-refs set; skipping arXiv metadata/domain filter", flush=True)
        metadata = {}
    else:
        metadata = fetch_metadata_for_ids(all_ref_ids, args)
        print(f"[refs] metadata rows fetched: {len(metadata)}", flush=True)

    # Exclude already-processed papers
    existing_ids: Set[str] = set()
    for d in (PROJECT_ROOT / "data" / "00_raw" / "mineru_output").glob("*"):
        if d.is_dir():
            existing_ids.add(d.name)
    for d in args.mineru_output_dir.glob("*"):
        if d.is_dir():
            existing_ids.add(d.name)

    rows: List[Dict[str, Any]] = []
    per_seed_kept: Dict[str, int] = defaultdict(int)
    row_seen: Set[str] = set()
    for seed in seed_refs:
        seed_meta = seed_rows.get(seed, {})
        refs = sorted(strip_version(r) for r in seed_refs[seed])
        ranked_refs = sorted(refs, key=lambda aid: (aid not in metadata, aid))
        for aid in ranked_refs:
            if per_seed_kept[seed] >= args.refs_per_seed:
                break
            if aid in row_seen:
                continue
            if aid in existing_ids or aid in seed_refs:
                continue
            meta = metadata.get(aid)
            if meta:
                # ── KEY DIFFERENCE from noncs version ──
                # Use is_huawei_domain() instead of is_noncs_categories()
                if not is_huawei_domain(meta.get("categories", [])):
                    continue
                categories = meta.get("categories", [])
                title = meta.get("title", "")
                year = meta.get("year")
            elif not args.allow_unverified_refs:
                continue
            else:
                categories = []
                title = ""
                year = None
            rows.append(
                {
                    "arxiv_id": aid,
                    "seed_arxiv_id": seed,
                    "seed_title": seed_meta.get("title", ""),
                    "seed_topic": seed_meta.get("seed_topic", ""),
                    "title": title,
                    "year": year,
                    "categories": categories,
                    "metadata_verified": bool(meta),
                }
            )
            row_seen.add(aid)
            per_seed_kept[seed] += 1

    # Round-robin selection across seeds for diversity
    selected: List[Dict[str, Any]] = []
    seed_order = sorted(seed_refs.keys(), key=lambda s: len(seed_refs.get(s, ())), reverse=True)
    by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed_arxiv_id"]].append(row)
    for seed in by_seed:
        by_seed[seed].sort(key=lambda r: (r.get("metadata_verified", False), r.get("year") or 0), reverse=True)

    round_idx = 0
    while len(selected) < args.candidate_limit:
        added = False
        for seed in seed_order:
            bucket = by_seed.get(seed, [])
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added = True
                if len(selected) >= args.candidate_limit:
                    break
        if not added:
            break
        round_idx += 1

    print(f"[refs] selected candidates: {len(selected)}", flush=True)
    print("[refs] per-seed selected counts:", flush=True)
    for seed in seed_order[: args.max_seeds]:
        cnt = sum(1 for r in selected if r["seed_arxiv_id"] == seed)
        if cnt:
            print(f"  {seed}: {cnt}", flush=True)
    return selected


# ── P5: Download target papers ─────────────────────────────────────────

def download_targets(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-huawei-target-download/1.0 (research)"})
    results: List[Dict[str, Any]] = []
    success_both = 0
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.latex_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for idx, row in enumerate(rows):
        aid = row["arxiv_id"]
        if success_both >= args.target:
            break
        if idx > 0 and idx % args.progress_interval == 0:
            elapsed = max(1.0, time.time() - start_time)
            print(
                f"[download] {idx}/{len(rows)} tried | success_both={success_both} | "
                f"rate={idx / elapsed * 3600:.1f}/hr | elapsed={elapsed / 60:.1f}min",
                flush=True,
            )

        time.sleep(args.download_delay)
        pdf_ok, pdf_msg = download_pdf(aid, args.pdf_dir, session)
        time.sleep(args.download_delay)
        latex_ok, latex_msg = download_latex(aid, args.latex_dir, session)
        out = {
            **row,
            "pdf_ok": pdf_ok,
            "pdf_msg": pdf_msg,
            "latex_ok": latex_ok,
            "latex_msg": latex_msg,
            "counted_success": bool(pdf_ok and latex_ok),
            "pdf_path": str(args.pdf_dir / f"{safe_name(aid)}.pdf"),
            "latex_extract_dir": str(args.latex_dir / "extracted" / safe_name(aid)),
        }
        results.append(out)
        if pdf_ok and latex_ok:
            success_both += 1
        if idx < 10 or (idx + 1) % args.progress_interval == 0:
            print(f"  {aid}: pdf={pdf_msg}; latex={latex_msg}; success={success_both}", flush=True)

    return results


# ── Summary ────────────────────────────────────────────────────────────

def summarize(
    seed_rows: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    success = [r for r in results if r.get("counted_success")]
    by_seed: Dict[str, int] = defaultdict(int)
    by_topic: Dict[str, int] = defaultdict(int)
    by_category: Dict[str, int] = defaultdict(int)
    for row in success:
        by_seed[row.get("seed_arxiv_id", "")] += 1
        by_topic[row.get("seed_topic", "")] += 1
        for cat in row.get("categories", []):
            by_category[cat] += 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_count": len(seed_rows),
        "candidate_count": len(candidates),
        "download_attempts": len(results),
        "success_both_pdf_latex": len(success),
        "pdf_ok": sum(1 for r in results if r.get("pdf_ok")),
        "latex_ok": sum(1 for r in results if r.get("latex_ok")),
        "by_seed_success": dict(sorted(by_seed.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_topic_success": dict(sorted(by_topic.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_categories_success": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))[:50]),
        "seed_rows": seed_rows,
        "candidates": candidates,
        "results": results,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--phase",
        choices=[
            "smoke", "search_surveys", "download_seeds", "expand_refs",
            "filter_domain", "download_refs", "all",
        ],
        default="search_surveys",
        help="Which pipeline phase to run (default: search_surveys)",
    )
    ap.add_argument("--target", type=int, default=3000,
                    help="Target papers with both PDF+LaTeX (default: 3000)")
    ap.add_argument("--refs-per-seed", type=int, default=30,
                    help="Max references to keep per review seed (default: 30)")
    ap.add_argument("--max-seeds", type=int, default=100,
                    help="Max review seeds to use (default: 100)")
    ap.add_argument("--max-seeds-per-topic", type=int, default=4,
                    help="Max seeds per topic query; 0 = no cap (default: 4)")
    ap.add_argument("--candidate-limit", type=int, default=5000,
                    help="Max candidates to pass to download phase (default: 5000)")
    ap.add_argument("--search-results-per-topic", type=int, default=60,
                    help="arXiv results per topic query (default: 60)")
    ap.add_argument("--api-delay", type=float, default=4.0)
    ap.add_argument("--api-retries", type=int, default=5)
    ap.add_argument("--download-delay", type=float, default=3.0)
    ap.add_argument("--progress-interval", type=int, default=20)
    ap.add_argument("--seed-source-dir", type=Path, default=DEFAULT_SEED_SOURCE_DIR)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    ap.add_argument("--mineru-output-dir", type=Path,
                    default=DEFAULT_BASE / "huawei_mineru_output")
    ap.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    ap.add_argument("--ids-output", type=Path, default=DEFAULT_TARGET_IDS)
    ap.add_argument("--success-ids-output", type=Path, default=DEFAULT_SUCCESS_IDS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--seed-id-file", type=Path, default=None)
    ap.add_argument("--seed-ids", nargs="*", default=None)
    ap.add_argument("--allow-unverified-refs", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    explicit_seed_ids = load_seed_ids(args)

    if args.phase == "smoke":
        args.search_results_per_topic = 5
        args.max_seeds = 15
        args.max_seeds_per_topic = 1
        seed_rows = search_review_seeds(args)
        seed_rows = seed_rows[: args.max_seeds]
        write_json(args.seed_manifest, {"seeds": seed_rows})
        write_ids(args.seed_manifest.with_suffix(".txt"), [r["arxiv_id"] for r in seed_rows])
        print(f"[seed] selected {len(seed_rows)} review seeds", flush=True)
        for i, row in enumerate(seed_rows, 1):
            print(f"  {i:02d}. {row['arxiv_id']} {row.get('categories', [])} {row.get('title', '')[:90]}", flush=True)
        return

    if args.phase == "download_only":
        if not args.ids_output.exists():
            raise FileNotFoundError(str(args.ids_output))
        candidates = [
            {"arxiv_id": line.strip()}
            for line in args.ids_output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seed_rows: List[Dict[str, Any]] = []
    else:
        if explicit_seed_ids:
            seed_rows = [
                {"arxiv_id": aid, "title": "", "categories": [], "seed_topic": "manual"}
                for aid in explicit_seed_ids
            ]
        else:
            seed_rows = search_review_seeds(args)
        seed_rows = seed_rows[: args.max_seeds]
        write_json(args.seed_manifest, {"seeds": seed_rows})
        write_ids(args.seed_manifest.with_suffix(".txt"), [r["arxiv_id"] for r in seed_rows])
        print(f"[seed] selected {len(seed_rows)} review seeds", flush=True)
        for i, row in enumerate(seed_rows[:20], 1):
            print(f"  {i:02d}. {row['arxiv_id']} {row.get('categories', [])} "
                  f"{row.get('title', '')[:90]}", flush=True)

        if args.phase == "search_surveys" or args.search_only:
            return

        seed_ids = [r["arxiv_id"] for r in seed_rows]
        seed_refs = collect_refs_from_seed_sources(seed_ids, args)
        seed_row_map = {r["arxiv_id"]: r for r in seed_rows}
        candidates = choose_target_candidates(seed_refs, seed_row_map, args)
        write_json(
            args.manifest.with_name(args.manifest.stem + "_candidates.json"),
            {"candidates": candidates},
        )
        write_ids(args.ids_output, [r["arxiv_id"] for r in candidates])

        if args.phase in ("expand_refs", "filter_domain") or args.collect_only:
            return

    results = download_targets(candidates, args)
    success_ids = [r["arxiv_id"] for r in results if r.get("counted_success")]
    write_ids(args.success_ids_output, success_ids)
    manifest = summarize(seed_rows, candidates, results)
    write_json(args.manifest, manifest)

    print("=" * 72, flush=True)
    print(f"Success with PDF+LaTeX: {len(success_ids)} / target {args.target}", flush=True)
    print(f"IDs:      {args.success_ids_output}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    print(f"PDF dir:  {args.pdf_dir}", flush=True)
    print(f"LaTeX:    {args.latex_dir}", flush=True)
    print("=" * 72, flush=True)
    if len(success_ids) < args.target:
        print(f"WARNING: Only {len(success_ids)} papers with both PDF+LaTeX; target={args.target}",
              flush=True)


if __name__ == "__main__":
    main()
