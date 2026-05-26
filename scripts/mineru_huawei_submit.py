#!/usr/bin/env python3
"""Submit all Huawei PDFs to MinerU API. No concurrency — sequential, resumable."""
import json, os, sys, time, requests
from pathlib import Path

PDF = Path("data/00_raw/huawei_pdfs")
OUT = Path("data/00_raw/huawei_mineru_output")
API = "http://localhost:8001/file_parse"
OUT.mkdir(parents=True, exist_ok=True)

pdfs = sorted(PDF.glob("*.pdf"))
done = {d.name for d in OUT.iterdir() if (d / f"{d.name}.md").exists()}
todo = [p for p in pdfs if p.stem not in done]

print(f"total={len(pdfs)} done={len(done)} todo={len(todo)}", flush=True)

n, fail = len(done), 0
t0 = time.time()
for i, pdf in enumerate(todo):
    aid = pdf.stem
    ok = False
    for attempt in range(3):
        try:
            with open(pdf, "rb") as f:
                r = requests.post(API,
                    files={"files": (f"{aid}.pdf", f)},
                    data={"parameters": json.dumps({"parse_method":"auto","lang":"en"})},
                    timeout=180)
            if r.status_code == 200:
                d = r.json()
                od = OUT / aid; od.mkdir(parents=True, exist_ok=True)
                (od / "mineru_result.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))
                for fn, c in d.get("results", {}).items():
                    if isinstance(c, dict) and "md_content" in c:
                        (od / f"{aid}.md").write_text(c["md_content"])
                n += 1; ok = True
                break
            elif r.status_code == 409:
                time.sleep(5)
            else:
                break
        except Exception as e:
            time.sleep(5)
    if not ok:
        fail += 1
        if fail <= 5:
            print(f"  FAIL {aid}", flush=True)
    if (i + 1) % 50 == 0:
        elap = (time.time() - t0) / 60
        print(f"  [{i+1}/{len(todo)}] ok={n} fail={fail} rate={int((i+1)/elap)}/min", flush=True)

print(f"done ok={n} fail={fail} total={n}", flush=True)
