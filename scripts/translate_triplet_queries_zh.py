#!/usr/bin/env python3
"""Translate triplet queries EN → ZH.

Only the `query` field is translated. The positive/negative passages stay in
English (the corpus is not translated — per user requirement). For
cross-lingual retrieval validation we additionally store a ZH→EN
back-translation, used downstream to measure semantic preservation.

Usage:
  python scripts/translate_triplet_queries_zh.py \
      --input  data/04_triplets/l1_dual_evidence_triplets_v2_pass.jsonl \
      --output data/04_triplets/l1_dual_evidence_triplets_v2_pass_zh.jsonl
"""

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_api_logger import wrap_requests_call
from src.utils.token_logger import log_run


PROMPT_EN_TO_ZH = """Translate this English research question into natural Simplified Chinese.

Hard rules:
1. KEEP IN ENGLISH (verbatim, do not transliterate, do not translate):
   - dataset names (e.g. Heritage Health, MovieLens, Reddit, Amazon Reviews, Law School, CIFAR, ImageNet)
   - model / method / algorithm names (e.g. BM25, BERT, CLAN, CESNA, FCRL, GAN, Transformer)
   - acronyms / metric names (e.g. STEM, CEO, F1, MRR, AUC, ROC, MSE, IoU)
   - product / tool / benchmark / institution names (proper nouns)
   - statistical / mathematical symbols (e.g. α, β, σ, p<0.05)
2. Common nouns SHOULD be translated normally (figure → 图, table → 表, accuracy → 准确率).
3. Output ONLY the Chinese translation. No explanation, no quotes.

English: {query}

Chinese:"""

PROMPT_ZH_TO_EN = """Translate this Chinese research question back into natural English.

Hard rules:
1. Any English token in the source (proper nouns, acronyms, model names) MUST appear in the output unchanged.
2. Preserve technical phrasing; keep statistical / mathematical symbols.
3. Output ONLY the English translation.

Chinese: {zh}

English:"""


def translate(api_key: str, api_url: str, model: str, prompt: str, max_retries: int = 3) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = wrap_requests_call(
                model=model,
                url=api_url,
                headers=headers,
                payload=payload,
                user="translate_triplet",
                timeout=60,
            )
            choice = resp["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            usage = resp.get("usage") or {}
            return text, usage
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"translate failed after {max_retries}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--no-backtranslate", action="store_true",
                    help="Skip the ZH→EN round-trip column (faster, half cost)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent worker threads. Default 1 (sequential).")
    ap.add_argument("--resume", action="store_true",
                    help="Append to output, skip query_ids/queries already present.")
    args = ap.parse_args()

    api_key = os.environ.get("COMPANY_API_KEY", "").strip()
    api_url = os.environ.get("COMPANY_API_URL", "").strip()
    if not (api_key and api_url):
        raise SystemExit("COMPANY_API_KEY / COMPANY_API_URL not set")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    triplets = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        triplets = triplets[:args.limit]
    print(f"Loaded {len(triplets)} triplets from {in_path}")

    # delivery_v1 uses `query_text`; older triplet_v2 schema uses `query`. Handle both.
    def _get_query(t: dict) -> str:
        return t.get("query_text") or t.get("query") or ""

    # Resume support: skip triplets whose query already appears in output.
    done_queries: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                qe = d.get("query_en") or _get_query(d)
                if qe:
                    done_queries.add(qe)
            except Exception:
                continue
        print(f"  Resume: skipping {len(done_queries)} already-translated rows")

    def _process(idx_t: tuple[int, dict]):
        idx, t = idx_t
        q_en = _get_query(t)
        if not q_en:
            return idx, None, 0, 0
        if q_en in done_queries:
            return idx, "__skip__", 0, 0
        try:
            q_zh, u1 = translate(api_key, api_url, args.model,
                                 PROMPT_EN_TO_ZH.format(query=q_en))
        except Exception as e:
            return idx, ("err_en2zh", str(e)), 0, 0
        in_t = u1.get("prompt_tokens", 0)
        out_t = u1.get("completion_tokens", 0)
        q_zh2en = ""
        if not args.no_backtranslate:
            try:
                q_zh2en, u2 = translate(api_key, api_url, args.model,
                                        PROMPT_ZH_TO_EN.format(zh=q_zh))
                in_t += u2.get("prompt_tokens", 0)
                out_t += u2.get("completion_tokens", 0)
            except Exception as e:
                # Forward translate succeeded; record what we have.
                return idx, {"q_zh": q_zh, "q_zh2en": "", "err": f"zh2en_failed: {e}"}, in_t, out_t
        return idx, {"q_zh": q_zh, "q_zh2en": q_zh2en, "q_en": q_en}, in_t, out_t

    total_in = total_out = 0
    fails = 0
    completed_lock = Lock()
    completed_n = 0
    mode = "a" if (args.resume and out_path.exists()) else "w"

    with out_path.open(mode, encoding="utf-8") as fout:
        write_lock = Lock()
        def emit(idx: int, t: dict, result_payload: dict):
            nonlocal completed_n
            out_row = {**t, "query_en": result_payload["q_en"],
                       "query_zh": result_payload["q_zh"],
                       "query_zh2en": result_payload["q_zh2en"]}
            with write_lock:
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()

        if args.workers <= 1:
            for idx, t in enumerate(triplets):
                idx, result, in_t, out_t = _process((idx, t))
                if isinstance(result, dict) and "q_zh" in result:
                    emit(idx, t, result)
                    total_in += in_t
                    total_out += out_t
                elif result == "__skip__":
                    pass
                else:
                    fails += 1
                completed_n += 1
                if completed_n % 50 == 0:
                    print(f"  [{completed_n}/{len(triplets)}] in={total_in} out={total_out} fails={fails}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_process, (i, t)): (i, t) for i, t in enumerate(triplets)}
                for fut in as_completed(futures):
                    i_orig, t_orig = futures[fut]
                    try:
                        idx, result, in_t, out_t = fut.result()
                    except Exception as e:
                        fails += 1
                        completed_n += 1
                        continue
                    with completed_lock:
                        completed_n += 1
                    if isinstance(result, dict) and "q_zh" in result:
                        emit(idx, t_orig, result)
                        total_in += in_t
                        total_out += out_t
                    elif result == "__skip__":
                        pass
                    else:
                        fails += 1
                    if completed_n % 50 == 0:
                        print(f"  [{completed_n}/{len(triplets)}] in={total_in} out={total_out} fails={fails}")

    print(f"\nDone. Wrote {out_path}")
    print(f"Tokens: in={total_in} out={total_out}  fails={fails}")

    log_run(
        script="translate_triplet_queries_zh",
        model=f"company:{args.model}",
        purpose="EN→ZH (+round-trip ZH→EN) translation of triplet queries for cross-lingual retrieval validation",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "input": str(in_path),
            "output": str(out_path),
            "n_triplets": len(triplets),
            "n_fails": fails,
            "round_trip": not args.no_backtranslate,
        },
    )


if __name__ == "__main__":
    main()
