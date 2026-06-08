#!/usr/bin/env python3
"""
Download accessible Huawei Cloud documentation pages.

Downloads:
  - Product index pages (intl/en-us + zh-cn): ~30 pages
  - Quickstart/getting-started pages where available
  - Product description overview pages

These are the static HTML portions; full doc trees are JS-rendered SPAs.

Usage:
  python scripts/download_huawei_cloud_docs.py
"""

import json, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/00_raw/huawei_cloud_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
})

# All known Huawei Cloud products with documentation
CLOUD_PRODUCTS = [
    ("ecs", "Elastic Cloud Server", "弹性云服务器"),
    ("cce", "Cloud Container Engine", "云容器引擎"),
    ("obs", "Object Storage Service", "对象存储服务"),
    ("rds", "Relational Database Service", "关系型数据库"),
    ("modelarts", "ModelArts AI Platform", "ModelArts AI开发平台"),
    ("gaussdb", "GaussDB Database", "GaussDB数据库"),
    ("dws", "Data Warehouse Service", "数据仓库服务"),
    ("dcs", "Distributed Cache Service", "分布式缓存服务"),
    ("elb", "Elastic Load Balance", "弹性负载均衡"),
    ("vpc", "Virtual Private Cloud", "虚拟私有云"),
    ("iam", "Identity and Access Management", "统一身份认证"),
    ("cts", "Cloud Trace Service", "云审计服务"),
    ("css", "Cloud Search Service", "云搜索服务"),
    ("cdm", "Cloud Data Migration", "云数据迁移"),
    ("dlf", "Data Lake Factory", "数据湖工厂"),
    ("mrs", "MapReduce Service", "MapReduce服务"),
    ("dds", "Document Database Service", "文档数据库服务"),
    ("smn", "Simple Message Notification", "消息通知服务"),
    ("functiongraph", "FunctionGraph", "函数工作流"),
    ("apig", "API Gateway", "API网关"),
    ("bcs", "Blockchain Service", "区块链服务"),
    ("iotda", "IoT Device Access", "IoT设备接入"),
    ("kafka", "Distributed Message Service", "分布式消息服务"),
    ("cloudtable", "CloudTable", "表格存储服务"),
    ("dis", "Data Ingestion Service", "数据接入服务"),
]


def download_page(url: str, dest: Path, timeout: int = 30) -> dict:
    """Download single page."""
    result = {"url": url, "filename": dest.name, "success": False, "size_bytes": 0, "error": None}

    if dest.exists() and dest.stat().st_size > 1000:
        result["success"] = True
        result["size_bytes"] = dest.stat().st_size
        result["status"] = "skipped"
        return result

    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        content = r.text
        if len(content) < 500:
            raise ValueError(f"Too small: {len(content)} bytes")
        dest.write_text(content, encoding="utf-8")
        result["success"] = True
        result["size_bytes"] = len(content)
    except Exception as e:
        result["error"] = str(e)[:150]
    return result


def main():
    tasks = []

    for prod_id, name_en, name_cn in CLOUD_PRODUCTS:
        # EN index page
        tasks.append({
            "url": f"https://support.huaweicloud.com/intl/en-us/{prod_id}/index.html",
            "product_id": prod_id,
            "product_name": name_en,
            "lang": "en",
            "page_type": "index",
            "filename": f"huawei_cloud_{prod_id}_index_en.html",
        })
        # CN index page  
        tasks.append({
            "url": f"https://support.huaweicloud.com/{prod_id}/index.html",
            "product_id": prod_id,
            "product_name": name_cn,
            "lang": "zh",
            "page_type": "index",
            "filename": f"huawei_cloud_{prod_id}_index_cn.html",
        })

    # Also add quickstart pages
    for prod_id in ["dws", "dcs", "css", "dds"]:
        tasks.append({
            "url": f"https://support.huaweicloud.com/intl/en-us/qs-{prod_id}/index.html",
            "product_id": prod_id,
            "product_name": f"{prod_id.upper()} Quick Start",
            "lang": "en",
            "page_type": "quickstart",
            "filename": f"huawei_cloud_{prod_id}_qs_en.html",
        })

    print(f"Downloading {len(tasks)} Huawei Cloud documentation pages...")

    results = []
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(download_page, t["url"], OUT_DIR / t["filename"]): t
            for t in tasks
        }
        for future in as_completed(future_map):
            t = future_map[future]
            try:
                r = future.result()
                r.update({"product_id": t["product_id"], "product_name": t["product_name"],
                          "lang": t["lang"], "page_type": t["page_type"]})
                results.append(r)
                if r.get("status") == "skipped":
                    skip += 1
                elif r["success"]:
                    ok += 1
                    print(f"  [ok] [{t['lang']}] {t['product_name'][:35]:35s} ({r['size_bytes']//1024}KB)")
                else:
                    fail += 1
                    print(f"  [fail] [{t['lang']}] {t['product_name'][:35]:35s} {r['error'][:50]}")
            except Exception as e:
                fail += 1
                print(f"  [err] {t['product_name']}: {e}")
            time.sleep(0.3)

    # Summary
    print(f"\n  Cloud Docs: {ok} ok, {skip} skip, {fail} fail")
    print(f"  Output: {OUT_DIR}/")

    # Manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "support.huaweicloud.com",
        "total": len(tasks),
        "success": ok,
        "failed": fail,
        "files": [r for r in results if r["success"]],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
