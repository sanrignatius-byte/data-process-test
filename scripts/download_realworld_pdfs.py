#!/usr/bin/env python3
"""Download 20 real-world PDFs: reports, manuals, whitepapers, datasheets.
These are NOT academic papers — used to stress-test MinerU graph building
on document types without LaTeX sources.

Sources:
  - SEC EDGAR: 10-K annual reports (Apple, Microsoft, NVIDIA)
  - GSMA/IETF/ITU: industry technical reports
  - Open-source: datasheets, reference manuals
  - Huawei: public whitepapers
"""

import os, time, requests
from pathlib import Path

OUT = Path("data/00_raw/realworld_pdfs")
OUT.mkdir(parents=True, exist_ok=True)

# ── Real-world PDFs from public URLs ──────────────────────────────────
PDFS = [
    # === 财报/年报 (financial reports) ===
    ("apple_10k_2024.pdf", "https://s2.q4cdn.com/470004039/files/doc_financials/2024/ar/_10-K-2024-(As-Filed).pdf"),
    ("microsoft_10k_2024.pdf", "https://microsoft.gcs-web.com/static-files/0b2cfdec-0afc-4f07-b8c1-71e3b7e64423"),
    ("nvidia_10k_2025.pdf", "https://s201.q4cdn.com/141608511/files/doc_financials/2025/ar/NVIDIA-2025-Annual-Report.pdf"),
    
    # === 技术标准/白皮书 (technical whitepapers) ===
    ("gsma_5g_spectrum.pdf", "https://www.gsma.com/spectrum/wp-content/uploads/2024/02/5G-Spectrum-Positions.pdf"),
    ("itu_ai4good_whitepaper.pdf", "https://www.itu.int/en/ITU-T/AI/Documents/ai4good-whitepaper.pdf"),
    ("ietf_rfc9110_http.pdf", "https://www.rfc-editor.org/rfc/rfc9110.pdf"),
    
    # === 技术手册 (technical manuals) ===
    ("riscv_isa_manual_v2.pdf", "https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf"),
    ("linux_kernel_docs.pdf", "https://www.kernel.org/doc/html/latest/pdf/linux-kernel.pdf"),
    ("postgresql_16_manual.pdf", "https://ftp.postgresql.org/pub/docs/postgresql-16-A4.pdf"),
    
    # === 产品说明书 (product specifications) ===
    ("nvidia_h100_datasheet.pdf", "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet"),
    ("apple_m3_pro_datasheet.pdf", "https://www.apple.com/macbook-pro/pdf/Apple_M3_Pro_Environmental_Report.pdf"),
    
    # === 华为公开文档 (Huawei public) ===
    ("huawei_6g_whitepaper.pdf", "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/2023/6g-architecture-whitepaper.pdf"),
    ("huawei_green_development.pdf", "https://www-file.huawei.com/-/media/corp2020/pdf/sustainability/huawei-2024-sustainability-report-en.pdf"),
    ("huawei_intelligent_world2030.pdf", "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/intelligent-world-2030-en.pdf"),
    
    # === 行业研报 (industry research) ===
    ("ieee_ai_trends_2024.pdf", "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=10521431"),
    ("worldbank_digital_dev.pdf", "https://documents1.worldbank.org/curated/en/099031924135041365/pdf/P1759711f693680131bcfb15141de9f9c5f.pdf"),
    
    # === 芯片/半导体文档 ===
    ("intel_xeon_spec.pdf", "https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/xeon-scalable-platform-brief.pdf"),
    ("arm_architecture_reference.pdf", "https://documentation-service.arm.com/static/64bb08acd90c3731d9b03327"),
    
    # === 开源项目文档 ===
    ("kubernetes_architecture.pdf", "https://kubernetes.io/docs/concepts/architecture/"),
    ("tensorflow_whitepaper.pdf", "https://arxiv.org/pdf/1603.04467"),
]

# ── Download ──────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; research-data-collection/1.0)",
})

ok, fail = 0, 0
for filename, url in PDFS:
    path = OUT / filename
    if path.exists() and path.stat().st_size > 10000:
        print(f"  [skip] {filename}")
        ok += 1
        continue
    
    try:
        r = session.get(url, timeout=60, stream=True, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 10000:
            path.write_bytes(r.content)
            size_kb = len(r.content) / 1024
            print(f"  [ok] {filename} ({size_kb:.0f}KB)")
            ok += 1
        else:
            print(f"  [fail] {filename}: HTTP {r.status_code}, size={len(r.content)}")
            fail += 1
    except Exception as e:
        print(f"  [err] {filename}: {e}")
        fail += 1
    
    time.sleep(2)  # be polite

print(f"\nDone: {ok} ok, {fail} fail")
print(f"Files in {OUT}: {len(list(OUT.glob('*.pdf')))}")
