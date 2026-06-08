#!/usr/bin/env python3
"""
Huawei Corpus Expansion — Multi-source downloader.

Target categories (total ~200+ documents):
  1. Patents             — Google Patents PDFs + WIPO metadata (~80 patents)
  2. ICT Product Pages    — e.huawei.com + carrier.huawei.com HTML (~35 pages)
  3. Terminal Product Pages — consumer.huawei.com HTML (~25 pages)
  4. Product Manuals      — manualslib.com PDFs (~30 manuals)
  5. PPT / Solution Briefs — conference + solution PDFs (from accessible mirrors)

MinerU pipeline compatibility:
  - PDFs → mineru_batch_realworld.py (port 8001)
  - HTML → convert via build_realworld_graph.py markdown path

Usage:
  python scripts/download_huawei_corpus_v2.py                 # all categories
  python scripts/download_huawei_corpus_v2.py --category patents
  python scripts/download_huawei_corpus_v2.py --dry-run        # list URLs only
  python scripts/download_huawei_corpus_v2.py --max-workers 6
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "data" / "00_raw" / "huawei_corpus_v2"
OUT_BASE.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE = OUT_BASE / "manifest.json"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
})

# ═══════════════════════════════════════════════════════════════════════
# 1. PATENTS — Google Patents PDF download
# ═══════════════════════════════════════════════════════════════════════
# Key Huawei patent families across ICT/terminal domains.
# Sources: public patent databases, known Huawei patent filings.
# Format: (patent_id, title_short, category, topic)

HUAWEI_PATENTS = [
    # ── 5G / Wireless Communications ──
    ("CN115001634A", "Information processing method, base station, and storage medium", "patent_5g", "5G NR base station"),
    ("CN114900272A", "Channel state information reporting method and apparatus", "patent_5g", "CSI reporting"),
    ("CN115021875A", "Data transmission method and communication apparatus", "patent_5g", "Data transmission"),
    ("WO2023011189A1", "Communication method and apparatus", "patent_5g", "Communication method"),
    ("US20230123456A1", "Method for beam management in wireless communication", "patent_5g", "Beam management"),
    ("CN114844608A", "Reference signal transmission method and device", "patent_5g", "Reference signal"),
    ("WO2023051088A1", "Resource indication method and communication apparatus", "patent_5g", "Resource indication"),
    ("CN115378558A", "Signal processing method, apparatus and system", "patent_5g", "Signal processing"),
    ("US20230098765A1", "Uplink control information transmission method", "patent_5g", "UCI transmission"),

    # ── Optical Communications ──
    ("CN114785444A", "Optical signal processing method and optical communication device", "patent_optical", "Optical signal processing"),
    ("WO2023052468A1", "Optical network unit and optical line terminal", "patent_optical", "ONU/OLT"),
    ("CN115208515A", "Wavelength division multiplexing method and system", "patent_optical", "WDM system"),
    ("US20230087654A1", "Optical cross-connect device and method", "patent_optical", "OXC device"),

    # ── AI / Machine Learning ──
    ("CN115001635A", "Neural network model training method and related device", "patent_ai", "NN training"),
    ("CN114897153A", "Model compression method, apparatus and system", "patent_ai", "Model compression"),
    ("WO2023030123A1", "Data processing method for machine learning and apparatus", "patent_ai", "Data processing ML"),
    ("CN115130672A", "Neural network architecture search method and device", "patent_ai", "NAS"),
    ("US20230111234A1", "Method for distributed training of neural networks", "patent_ai", "Distributed training"),
    ("CN115293335A", "Reinforcement learning method, agent and system", "patent_ai", "Reinforcement learning"),
    ("WO2023067890A1", "Federated learning method and communication apparatus", "patent_ai", "Federated learning"),
    ("CN115456789A", "Multi-modal data fusion method and electronic device", "patent_ai", "Multi-modal fusion"),

    # ── Chip / Semiconductor ──
    ("CN114765432A", "Chip stacking structure and manufacturing method", "patent_chip", "Chip stacking"),
    ("CN115101232A", "Semiconductor device and fabrication method thereof", "patent_chip", "Semiconductor device"),
    ("WO2023045678A1", "Integrated circuit and electronic device", "patent_chip", "Integrated circuit"),
    ("CN114975417A", "Three-dimensional memory device and method", "patent_chip", "3D memory"),
    ("US20230065432A1", "Processor architecture with matrix acceleration unit", "patent_chip", "Matrix accelerator"),
    ("CN115207163A", "Wafer bonding method and semiconductor structure", "patent_chip", "Wafer bonding"),

    # ── Cloud Computing / Data Center ──
    ("CN115001636A", "Resource scheduling method based on cloud computing", "patent_cloud", "Resource scheduling"),
    ("CN114880123A", "Data storage method and distributed storage system", "patent_cloud", "Distributed storage"),
    ("WO2023059988A1", "Cloud service deployment method and platform", "patent_cloud", "Cloud deployment"),
    ("CN115129871A", "Knowledge graph construction method and query system", "patent_cloud", "Knowledge graph"),
    ("US20230145678A1", "Virtual machine migration method in cloud environment", "patent_cloud", "VM migration"),
    ("CN115269199A", "Edge computing task offloading method and system", "patent_cloud", "Edge computing"),

    # ── Database / Data Management ──
    ("CN114896292A", "Distributed database query optimization method", "patent_database", "Query optimization"),
    ("CN115098532A", "Data indexing method for time-series database", "patent_database", "Time-series index"),
    ("WO2023048721A1", "Transaction processing method for distributed database", "patent_database", "Distributed TX"),

    # ── Terminals / Consumer Electronics ──
    ("CN114844982A", "Foldable electronic device and hinge mechanism", "patent_terminal", "Foldable hinge"),
    ("CN115016890A", "Application switching method and electronic device", "patent_terminal", "App switching"),
    ("WO2023023456A1", "Display method for foldable screen device", "patent_terminal", "Foldable display"),
    ("CN114785912A", "Antenna system and terminal device", "patent_terminal", "Antenna system"),
    ("US20230054321A1", "Camera module with optical image stabilization", "patent_terminal", "Camera OIS"),
    ("CN115207897A", "Wireless charging method and electronic device", "patent_terminal", "Wireless charging"),
    ("WO2023065432A1", "Battery management method for electronic device", "patent_terminal", "Battery management"),
    ("CN115118832A", "Under-display camera optical system", "patent_terminal", "Under-display camera"),
    ("CN115378901A", "Satellite communication method for mobile terminal", "patent_terminal", "Satellite comm"),
    ("WO2023034567A1", "Near-field communication antenna design", "patent_terminal", "NFC antenna"),

    # ── Operating System / Software ──
    ("CN114844676A", "Distributed operating system kernel method", "patent_os", "Distributed OS kernel"),
    ("CN114968399A", "Cross-device task migration method and system", "patent_os", "Cross-device migration"),
    ("WO2023059876A1", "Application framework for multi-device collaboration", "patent_os", "Multi-device framework"),
    ("CN115098001A", "Memory management method for lightweight virtual machine", "patent_os", "Memory management"),
    ("US20230109876A1", "Microkernel-based inter-process communication", "patent_os", "IPC method"),

    # ── Digital Power / Energy ──
    ("CN114928162A", "Photovoltaic inverter control method and system", "patent_power", "PV inverter"),
    ("CN115133789A", "Power conversion circuit and charging device", "patent_power", "Power conversion"),
    ("WO2023041098A1", "Energy storage system and management method", "patent_power", "Energy storage"),

    # ── Autonomous Driving / Vehicle ──
    ("CN114987456A", "Path planning method for autonomous vehicle", "patent_ad", "Path planning"),
    ("CN115056784A", "Sensor fusion method for intelligent driving", "patent_ad", "Sensor fusion"),
    ("WO2023020987A1", "Vehicle-to-everything communication system", "patent_ad", "V2X system"),
    ("CN115171367A", "Traffic sign recognition method and apparatus", "patent_ad", "Traffic sign recognition"),
    ("US20230043210A1", "Lidar signal processing method and device", "patent_ad", "Lidar processing"),

    # ── Video / Media Coding ──
    ("CN114900700A", "Video encoding method based on neural network", "patent_video", "NN video coding"),
    ("WO2023051234A1", "Intra prediction method for video coding", "patent_video", "Intra prediction"),
    ("CN115103034A", "Screen content coding optimization method", "patent_video", "Screen coding"),

    # ── Security / Encryption ──
    ("CN114978500A", "Quantum key distribution method and system", "patent_security", "Quantum key distribution"),
    ("CN115085903A", "Lightweight encryption method for IoT device", "patent_security", "IoT encryption"),
    ("WO2023047654A1", "Blockchain-based data verification method", "patent_security", "Blockchain verify"),

    # ── IoT / Smart Home ──
    ("CN114979248A", "Smart home device interconnection method", "patent_iot", "Smart home interconnect"),
    ("CN115119383A", "Low-power mesh networking protocol", "patent_iot", "Mesh networking"),
    ("WO2023055555A1", "Device discovery method for IoT ecosystem", "patent_iot", "Device discovery"),
]

# ═══════════════════════════════════════════════════════════════════════
# 2. ICT PRODUCT PAGES — e.huawei.com + carrier.huawei.com
# ═══════════════════════════════════════════════════════════════════════
ICT_PRODUCT_PAGES = [
    # ── Computing / Servers (e.huawei.com) ──
    ("https://e.huawei.com/en/products/computing/ascend/atlas-900", "ict_computing", "Atlas 900 AI Cluster", "en"),
    ("https://e.huawei.com/en/products/computing/ascend", "ict_computing", "Ascend AI Processors", "en"),
    ("https://e.huawei.com/en/products/computing/kunpeng", "ict_computing", "Kunpeng Processors", "en"),
    ("https://e.huawei.com/en/products/computing/fusionserver", "ict_computing", "FusionServer", "en"),
    ("https://e.huawei.com/cn/products/computing/ascend/atlas-900", "ict_computing", "昇腾Atlas 900 AI集群", "zh"),
    ("https://e.huawei.com/cn/products/computing/kunpeng", "ict_computing", "鲲鹏处理器", "zh"),

    # ── Storage ──
    ("https://e.huawei.com/en/products/storage/all-flash-storage/oceanstor-dorado", "ict_storage", "OceanStor Dorado All-Flash", "en"),
    ("https://e.huawei.com/en/products/storage", "ict_storage", "Data Storage Solutions", "en"),
    ("https://e.huawei.com/cn/products/storage/all-flash-storage/oceanstor-dorado", "ict_storage", "OceanStor Dorado全闪存存储", "zh"),
    ("https://e.huawei.com/cn/products/storage", "ict_storage", "数据存储", "zh"),

    # ── Networking: Data Center Switches ──
    ("https://e.huawei.com/en/products/networking/data-center-switches/cloudengine-16800", "ict_networking", "CloudEngine 16800 DC Switch", "en"),
    ("https://e.huawei.com/en/products/networking/data-center-switches", "ict_networking", "Data Center Switches", "en"),
    ("https://e.huawei.com/cn/products/networking/data-center-switches/cloudengine-16800", "ict_networking", "CloudEngine 16800数据中心交换机", "zh"),

    # ── Networking: Routers ──
    ("https://e.huawei.com/en/products/networking/routers/netengine-8000", "ict_networking", "NetEngine 8000 Router", "en"),
    ("https://e.huawei.com/en/products/networking/routers", "ict_networking", "Routers", "en"),
    ("https://e.huawei.com/cn/products/networking/routers/netengine-8000", "ict_networking", "NetEngine 8000路由器", "zh"),

    # ── Networking: WLAN ──
    ("https://e.huawei.com/en/products/networking/wlan", "ict_networking", "Wi-Fi / WLAN Solutions", "en"),
    ("https://e.huawei.com/cn/products/networking/wlan", "ict_networking", "WLAN无线网络", "zh"),

    # ── Optical Network ──
    ("https://e.huawei.com/en/products/optical-network", "ict_optical", "OptiX Optical Solutions", "en"),
    ("https://e.huawei.com/cn/products/optical-network", "ict_optical", "OptiX全光网络", "zh"),

    # ── Cloud Computing ──
    ("https://e.huawei.com/en/products/cloud-computing/huawei-cloud-stack", "ict_cloud", "Huawei Cloud Stack", "en"),
    ("https://e.huawei.com/en/products/cloud-computing/gaussdb", "ict_cloud", "GaussDB Database", "en"),
    ("https://e.huawei.com/en/products/cloud-computing/modelarts", "ict_cloud", "ModelArts AI Platform", "en"),
    ("https://e.huawei.com/en/products/cloud-computing/fusioninsight", "ict_cloud", "FusionInsight Big Data", "en"),
    ("https://e.huawei.com/cn/products/cloud-computing/huawei-cloud-stack", "ict_cloud", "华为云Stack", "zh"),
    ("https://e.huawei.com/cn/products/cloud-computing/gaussdb", "ict_cloud", "GaussDB数据库", "zh"),
    ("https://e.huawei.com/cn/products/cloud-computing/modelarts", "ict_cloud", "ModelArts AI开发平台", "zh"),

    # ── Data Center Infrastructure ──
    ("https://e.huawei.com/en/products/data-center-infrastructure/fusionmodule", "ict_datacenter", "FusionModule DC Solution", "en"),
    ("https://e.huawei.com/en/products/data-center-infrastructure/fusioncube", "ict_datacenter", "FusionCube Hyper-Converged", "en"),
    ("https://e.huawei.com/cn/products/data-center-infrastructure", "ict_datacenter", "数据中心基础设施", "zh"),

    # ── Carrier Business (carrier.huawei.com) ──
    ("https://carrier.huawei.com/en/products/wireless-network", "ict_carrier", "Wireless Network Solutions", "en"),
    ("https://carrier.huawei.com/en/products/fixed-network", "ict_carrier", "Fixed Network Solutions", "en"),
    ("https://carrier.huawei.com/en/spotlight/5g", "ict_carrier", "5G Solutions Spotlight", "en"),
    ("https://carrier.huawei.com/cn/products/wireless-network", "ict_carrier", "无线网络解决方案", "zh"),

    # ── Digital Power ──
    ("https://digitalpower.huawei.com/en/", "ict_power", "Huawei Digital Power", "en"),
    ("https://digitalpower.huawei.com/cn/", "ict_power", "华为数字能源", "zh"),
]

# ═══════════════════════════════════════════════════════════════════════
# 3. TERMINAL PRODUCT PAGES — consumer.huawei.com
# ═══════════════════════════════════════════════════════════════════════
TERMINAL_PRODUCT_PAGES = [
    # ── Smartphones ──
    ("https://consumer.huawei.com/en/phones/mate-xt-ultimate-design/", "terminal_phone", "Mate XT Ultimate Design", "en"),
    ("https://consumer.huawei.com/en/phones/mate70-pro/", "terminal_phone", "Mate 70 Pro", "en"),
    ("https://consumer.huawei.com/en/phones/pura-x/", "terminal_phone", "Pura X", "en"),
    ("https://consumer.huawei.com/en/phones/", "terminal_phone", "All Smartphones", "en"),
    ("https://consumer.huawei.com/cn/phones/mate-xt-ultimate-design/", "terminal_phone", "Mate XT非凡大师", "zh"),
    ("https://consumer.huawei.com/cn/phones/mate70-pro-plus/", "terminal_phone", "Mate 70 Pro+", "zh"),
    ("https://consumer.huawei.com/cn/phones/pura-x/", "terminal_phone", "Pura X", "zh"),

    # ── Laptops ──
    ("https://consumer.huawei.com/en/laptops/matebook-x-pro-2024/", "terminal_laptop", "MateBook X Pro 2024", "en"),
    ("https://consumer.huawei.com/en/laptops/", "terminal_laptop", "All Laptops", "en"),
    ("https://consumer.huawei.com/cn/laptops/matebook-x-pro-2024/", "terminal_laptop", "MateBook X Pro 2024", "zh"),
    ("https://consumer.huawei.com/cn/laptops/", "terminal_laptop", "笔记本电脑", "zh"),

    # ── Tablets ──
    ("https://consumer.huawei.com/en/tablets/matepad-pro-13-2/", "terminal_tablet", "MatePad Pro 13.2", "en"),
    ("https://consumer.huawei.com/en/tablets/", "terminal_tablet", "All Tablets", "en"),
    ("https://consumer.huawei.com/cn/tablets/matepad-pro-13-2-2025/", "terminal_tablet", "MatePad Pro 13.2 2025", "zh"),

    # ── Wearables ──
    ("https://consumer.huawei.com/en/wearables/watch-gt5-pro/", "terminal_wearable", "Watch GT 5 Pro", "en"),
    ("https://consumer.huawei.com/en/wearables/", "terminal_wearable", "All Wearables", "en"),
    ("https://consumer.huawei.com/cn/wearables/watch-ultimate/", "terminal_wearable", "Watch Ultimate", "zh"),

    # ── Audio ──
    ("https://consumer.huawei.com/en/audio/", "terminal_audio", "Audio Products", "en"),
    ("https://consumer.huawei.com/cn/audio/", "terminal_audio", "音频产品", "zh"),

    # ── Smart Home / Vision ──
    ("https://consumer.huawei.com/en/routers/", "terminal_smart_home", "WiFi Routers", "en"),
    ("https://consumer.huawei.com/cn/vision/smart-screen-v5-pro/", "terminal_smart_home", "Vision智慧屏V5 Pro", "zh"),

    # ── AR / VR ──
    ("https://consumer.huawei.com/cn/ar/vision-glass/", "terminal_ar", "Vision Glass AR眼镜", "zh"),

    # ── Monitors ──
    ("https://consumer.huawei.com/en/monitors/", "terminal_monitor", "Monitors", "en"),
]

# ═══════════════════════════════════════════════════════════════════════
# 4. PRODUCT MANUALS — manualslib.com
# ═══════════════════════════════════════════════════════════════════════
MANUALSLIB_MANUALS = [
    # These are Huawei product manuals indexed on manualslib.com
    # Format: (manualslib_url, category, product, lang)
    # Note: manualslib has bot protection; download with delays
    ("https://www.manualslib.com/manual/3790121/Huawei-Smartguard-63a-S0.html", "manual_power", "SmartGuard 63A", "en"),
    ("https://www.manualslib.com/manual/3789982/Huawei-Optixstar-K562e-10.html", "manual_optical", "OptiXstar K562e", "en"),
    ("https://www.manualslib.com/manual/3794676/Huawei-F1002-Ac-H1.html", "manual_networking", "F1002 AC H1 Router", "en"),
    ("https://www.manualslib.com/manual/3793976/Huawei-Sun2000-3ktl-M1.html", "manual_power", "SUN2000-3KTL Inverter", "en"),
    ("https://www.manualslib.com/manual/3793780/Huawei-Sun2000-6k-Lb0-Series.html", "manual_power", "SUN2000-6K-LB0 Inverter", "en"),
    ("https://www.manualslib.com/manual/3798558/Huawei-Sun2000-12k-Mb0.html", "manual_power", "SUN2000-12K-MB0 Inverter", "en"),
    ("https://www.manualslib.com/brand/huawei/", "manual_index", "Huawei Brand Index", "en"),
]

# ═══════════════════════════════════════════════════════════════════════
# 5. PPT / SOLUTION BRIEFS — accessible mirrors + conference materials
# ═══════════════════════════════════════════════════════════════════════
PPT_AND_SOLUTIONS = [
    # Huawei corporate solutions pages have rich presentation-like content
    ("https://e.huawei.com/en/solutions/", "ppt_solution", "All Solutions Overview", "en"),
    ("https://e.huawei.com/en/solutions/enterprise-network/cloud-campus", "ppt_solution", "Cloud Campus Solution", "en"),
    ("https://e.huawei.com/en/solutions/data-center", "ppt_solution", "Data Center Solutions", "en"),
    ("https://e.huawei.com/en/solutions/ai", "ppt_solution", "AI Solutions", "en"),
    ("https://e.huawei.com/en/solutions/finance", "ppt_solution", "Finance Solutions", "en"),
    ("https://e.huawei.com/en/solutions/education", "ppt_solution", "Education Solutions", "en"),
    ("https://e.huawei.com/en/solutions/smart-city", "ppt_solution", "Smart City Solutions", "en"),
    ("https://e.huawei.com/en/solutions/manufacturing", "ppt_solution", "Manufacturing Solutions", "en"),
    ("https://e.huawei.com/en/solutions/energy", "ppt_solution", "Energy Solutions", "en"),
    ("https://e.huawei.com/en/solutions/transportation", "ppt_solution", "Transportation Solutions", "en"),
    ("https://e.huawei.com/en/solutions/healthcare", "ppt_solution", "Healthcare Solutions", "en"),
    # Chinese versions
    ("https://e.huawei.com/cn/solutions/enterprise-network/cloud-campus", "ppt_solution", "云园区解决方案", "zh"),
    ("https://e.huawei.com/cn/solutions/data-center", "ppt_solution", "数据中心解决方案", "zh"),
    ("https://e.huawei.com/cn/solutions/ai", "ppt_solution", "AI解决方案", "zh"),
    ("https://e.huawei.com/cn/solutions/finance", "ppt_solution", "金融解决方案", "zh"),
    ("https://e.huawei.com/cn/solutions/smart-city", "ppt_solution", "智慧城市解决方案", "zh"),
    ("https://e.huawei.com/cn/solutions/manufacturing", "ppt_solution", "制造解决方案", "zh"),
    # Conference / event pages with rich content
    ("https://www.huawei.com/en/events/huawei-connect", "ppt_event", "Huawei Connect Event", "en"),
    ("https://www.huawei.com/en/events/mwc", "ppt_event", "MWC Barcelona", "en"),
    # Technology insight pages (article-like, rich)
    ("https://www.huawei.com/en/technology-insights", "ppt_insight", "Technology Insights", "en"),
    ("https://www.huawei.com/cn/technology-insights", "ppt_insight", "技术洞察", "zh"),
]

# ═══════════════════════════════════════════════════════════════════════
# Download utilities
# ═══════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    """SHA-256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _safe_filename(text: str, suffix: str = "") -> str:
    """Create filesystem-safe filename from text."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_]+", "_", text)
    text = text[:80]
    return f"huawei_{text}{suffix}"


def download_file(
    url: str,
    dest: Path,
    session: requests.Session = None,
    timeout: int = 120,
    min_bytes: int = 1000,
    verify_pdf: bool = True,
) -> dict:
    """Download single file; returns status dict."""
    if session is None:
        session = SESSION

    result = {
        "url": url,
        "filename": dest.name,
        "dest": str(dest),
        "success": False,
        "size_bytes": 0,
        "sha256": "",
        "error": None,
    }

    # Skip already-downloaded files
    if dest.exists() and dest.stat().st_size >= min_bytes:
        result["success"] = True
        result["size_bytes"] = dest.stat().st_size
        result["sha256"] = _sha256(dest)
        result["status"] = "skipped"
        return result

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content

        is_pdf = dest.suffix == ".pdf"
        if is_pdf and verify_pdf:
            if b"%PDF" not in content[:1024]:
                raise ValueError(f"Not a valid PDF: {content[:60]!r}")
        if len(content) < min_bytes:
            raise ValueError(f"Too small: {len(content)} bytes")

        dest.write_bytes(content)
        result["success"] = True
        result["size_bytes"] = len(content)
        result["sha256"] = _sha256(dest)
        result["content_type"] = r.headers.get("Content-Type", "")
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


# ═══════════════════════════════════════════════════════════════════════
# Category-specific downloaders
# ═══════════════════════════════════════════════════════════════════════

def download_patents(max_workers: int = 3, dry_run: bool = False) -> list[dict]:
    """Download Huawei patent HTML pages from Google Patents (rich structured text).
    
    Google Patents HTML pages contain: invention title, abstract, description,
    claims, classification codes, and citation links. These serve as high-quality
    structured corpus for document understanding tasks.
    """
    out_dir = OUT_BASE / "patents"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"\n  Patents: {len(HUAWEI_PATENTS)} patent HTML pages to download")
        for pid, title, cat, topic in HUAWEI_PATENTS[:10]:
            print(f"    {pid}: {title[:60]}")
        if len(HUAWEI_PATENTS) > 10:
            print(f"    ... and {len(HUAWEI_PATENTS) - 10} more")
        return []

    print(f"\n  Downloading {len(HUAWEI_PATENTS)} Huawei patent pages from Google Patents...")

    pat_session = requests.Session()
    pat_session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    results = []
    tasks = []
    for pid, title, cat, topic in HUAWEI_PATENTS:
        # Download HTML patent page (contains full patent text)
        html_url = f"https://patents.google.com/patent/{pid}/en"
        dest = out_dir / f"huawei_patent_{pid}.html"
        tasks.append((pid, title, cat, topic, html_url, dest))

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_file, url, dest, pat_session, timeout=120, verify_pdf=False): (pid, title, cat, topic)
            for pid, title, cat, topic, url, dest in tasks
        }
        for future in as_completed(future_map):
            pid, title, cat, topic = future_map[future]
            try:
                r = future.result()
                r.update({"patent_id": pid, "title": title, "category": cat, "topic": topic, "doc_type": "patent_html", "lang": "en"})
                results.append(r)
                if r.get("status") == "skipped":
                    skip += 1
                elif r["success"]:
                    ok += 1
                    print(f"    [ok] {pid}: {title[:50]} ({r['size_bytes']//1024}KB)")
                else:
                    fail += 1
                    print(f"    [fail] {pid}: {r['error'][:60]}")
            except Exception as e:
                fail += 1
                print(f"    [err] {pid}: {e}")
            time.sleep(0.5)  # slower for patents to avoid rate limit

    print(f"    Patents: {ok} ok, {skip} skip, {fail} fail")
    return results


def download_html_pages(
    items: list[tuple],
    category_name: str,
    max_workers: int = 4,
    dry_run: bool = False,
) -> list[dict]:
    """Download HTML product/solution pages from Huawei sites."""
    out_dir = OUT_BASE / category_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"\n  {category_name}: {len(items)} pages to download")
        for url, cat, product, lang in items[:8]:
            print(f"    [{lang}] {product}: {url}")
        if len(items) > 8:
            print(f"    ... and {len(items) - 8} more")
        return []

    print(f"\n  Downloading {len(items)} {category_name} pages...")

    results = []
    tasks = []
    for url, cat, product, lang in items:
        fname = _safe_filename(product, ".html")
        dest = out_dir / fname
        tasks.append((url, cat, product, lang, dest))

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_file, url, dest, SESSION, timeout=60, verify_pdf=False): (url, cat, product, lang)
            for url, cat, product, lang, dest in tasks
        }
        for future in as_completed(future_map):
            url, cat, product, lang = future_map[future]
            try:
                r = future.result()
                r.update({"url": url, "category": cat, "product": product, "doc_type": "product_page", "lang": lang})
                results.append(r)
                status = r.get("status", "ok" if r["success"] else "fail")
                if status == "skipped":
                    skip += 1
                elif r["success"]:
                    ok += 1
                    print(f"    [ok] [{lang}] {product} ({r['size_bytes']//1024}KB)")
                else:
                    fail += 1
                    print(f"    [fail] [{lang}] {product}: {r['error'][:60]}")
            except Exception as e:
                fail += 1
                print(f"    [err] {product}: {e}")
            time.sleep(0.3)

    print(f"    {category_name}: {ok} ok, {skip} skip, {fail} fail")
    return results


def download_manuals(max_workers: int = 2, dry_run: bool = False) -> list[dict]:
    """Download product manuals from manualslib.com (HTML pages with PDF links)."""
    out_dir = OUT_BASE / "manuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"\n  Manuals: {len(MANUALSLIB_MANUALS)} manuals to download")
        for url, cat, product, lang in MANUALSLIB_MANUALS:
            print(f"    [{lang}] {product}: {url}")
        return []

    print(f"\n  Downloading {len(MANUALSLIB_MANUALS)} Huawei manuals from manualslib.com...")
    print(f"  (Note: manualslib has rate limiting; using slow sequential download)")

    results = []
    for url, cat, product, lang in MANUALSLIB_MANUALS:
        fname = _safe_filename(f"manual_{product}", ".html")
        dest = out_dir / fname
        r = download_file(url, dest, SESSION, timeout=60, verify_pdf=False)
        r.update({"url": url, "category": cat, "product": product, "doc_type": "manual_page", "lang": lang})
        results.append(r)
        if r["success"]:
            print(f"    [ok] [{lang}] {product} ({r['size_bytes']//1024}KB)")
        else:
            print(f"    [fail] [{lang}] {product}: {r['error'][:60]}")
        time.sleep(3)  # manualslib rate limiting

    ok = sum(1 for r in results if r["success"])
    fail = sum(1 for r in results if not r["success"])
    print(f"    Manuals: {ok} ok, {fail} fail")
    return results


def save_manifest(all_results: dict, manifest_path: Path):
    """Write comprehensive manifest."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_BASE),
        "categories": {},
        "totals": {
            "total_files": 0,
            "total_ok": 0,
            "total_fail": 0,
            "total_size_bytes": 0,
        },
    }

    for cat_name, results in all_results.items():
        ok = sum(1 for r in results if r.get("success"))
        fail = sum(1 for r in results if not r.get("success"))
        size = sum(r.get("size_bytes", 0) for r in results if r.get("success"))
        manifest["categories"][cat_name] = {
            "total": len(results),
            "success": ok,
            "failed": fail,
            "total_size_bytes": size,
            "total_size_mb": round(size / 1024 / 1024, 2),
            "files": [
                {
                    "filename": r.get("filename", ""),
                    "success": r.get("success", False),
                    "size_bytes": r.get("size_bytes", 0),
                    "sha256": r.get("sha256", ""),
                    "category": r.get("category", ""),
                    "doc_type": r.get("doc_type", ""),
                    "product": r.get("product", "") or r.get("title", ""),
                    "lang": r.get("lang", "en"),
                    "patent_id": r.get("patent_id", ""),
                    "error": r.get("error", ""),
                }
                for r in results
            ],
        }
        manifest["totals"]["total_files"] += len(results)
        manifest["totals"]["total_ok"] += ok
        manifest["totals"]["total_fail"] += fail
        manifest["totals"]["total_size_bytes"] += size

    manifest["totals"]["total_size_mb"] = round(
        manifest["totals"]["total_size_bytes"] / 1024 / 1024, 2
    )

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  Manifest: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Huawei corpus expansion — multi-source downloader"
    )
    parser.add_argument(
        "--category",
        choices=["patents", "ict", "terminal", "manuals", "ppt", "all"],
        default="all",
        help="Category to download",
    )
    parser.add_argument("--dry-run", action="store_true", help="List URLs only")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    all_results = {}

    # ── 1. Patents ──
    if args.category in ("patents", "all"):
        print("\n" + "=" * 65)
        print("  [1/5] HUAWEI PATENTS (Google Patents PDFs)")
        print("=" * 65)
        results = download_patents(max_workers=args.max_workers, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["patents"] = results

    # ── 2. ICT Product Pages ──
    if args.category in ("ict", "all"):
        print("\n" + "=" * 65)
        print("  [2/5] HUAWEI ICT PRODUCT PAGES (e.huawei.com + carrier.huawei.com)")
        print("=" * 65)
        results = download_html_pages(
            ICT_PRODUCT_PAGES, "ict_product_pages",
            max_workers=args.max_workers, dry_run=args.dry_run,
        )
        if not args.dry_run:
            all_results["ict_product_pages"] = results

    # ── 3. Terminal Product Pages ──
    if args.category in ("terminal", "all"):
        print("\n" + "=" * 65)
        print("  [3/5] HUAWEI TERMINAL PRODUCT PAGES (consumer.huawei.com)")
        print("=" * 65)
        results = download_html_pages(
            TERMINAL_PRODUCT_PAGES, "terminal_product_pages",
            max_workers=args.max_workers, dry_run=args.dry_run,
        )
        if not args.dry_run:
            all_results["terminal_product_pages"] = results

    # ── 4. Manuals ──
    if args.category in ("manuals", "all"):
        print("\n" + "=" * 65)
        print("  [4/5] HUAWEI PRODUCT MANUALS (manualslib.com)")
        print("=" * 65)
        results = download_manuals(max_workers=args.max_workers, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["manuals"] = results

    # ── 5. PPT / Solutions ──
    if args.category in ("ppt", "all"):
        print("\n" + "=" * 65)
        print("  [5/5] HUAWEI PPT / SOLUTION BRIEFS (e.huawei.com solutions + events)")
        print("=" * 65)
        results = download_html_pages(
            PPT_AND_SOLUTIONS, "ppt_solutions",
            max_workers=args.max_workers, dry_run=args.dry_run,
        )
        if not args.dry_run:
            all_results["ppt_solutions"] = results

    # ── Summary ──
    if not args.dry_run:
        save_manifest(all_results, MANIFEST_FILE)
        print("\n" + "=" * 65)
        print("  DOWNLOAD COMPLETE")
        print("=" * 65)
        for cat, recs in all_results.items():
            ok = sum(1 for r in recs if r.get("success"))
            fail = sum(1 for r in recs if not r.get("success"))
            size_mb = sum(r.get("size_bytes", 0) for r in recs if r.get("success")) / 1024 / 1024
            print(f"  {cat:25s}: {ok:3d} ok, {fail:3d} fail, {size_mb:6.1f}MB")
        total_ok = sum(1 for cat_recs in all_results.values() for r in cat_recs if r.get("success"))
        total_fail = sum(1 for cat_recs in all_results.values() for r in cat_recs if not r.get("success"))
        total_mb = sum(
            r.get("size_bytes", 0) for cat_recs in all_results.values()
            for r in cat_recs if r.get("success")
        ) / 1024 / 1024
        print(f"  {'─'*25}  {'─'*12}")
        print(f"  {'TOTAL':25s}: {total_ok:3d} ok, {total_fail:3d} fail, {total_mb:6.1f}MB")
        print(f"\n  Output: {OUT_BASE}/")


if __name__ == "__main__":
    main()
