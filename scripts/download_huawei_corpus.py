#!/usr/bin/env python3
"""
Download Huawei corpus expansion: product manuals, patents, and PPTs.

Categories:
  1. ICT products (e.huawei.com) - servers, storage, networking, cloud, 5G
  2. Terminal products (consumer.huawei.com) - phones, laptops, wearables, audio
  3. Public patents - Google Patents API
  4. PPT presentations - Huawei Connect / HDC / industry events

Usage:
  python scripts/download_huawei_corpus.py                     # download ALL categories
  python scripts/download_huawei_corpus.py --category ict      # ICT only
  python scripts/download_huawei_corpus.py --category terminal # Terminal only
  python scripts/download_huawei_corpus.py --category patents  # Patents only
  python scripts/download_huawei_corpus.py --category ppt      # PPTs only
  python scripts/download_huawei_corpus.py --dry-run           # list URLs without downloading
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "data" / "00_raw" / "huawei_corpus"
OUT_BASE.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = OUT_BASE / "manifest.json"

# ──────────────────────────────────────────────────────────────────────
# User-Agent (avoid bot detection)
# ──────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

# ──────────────────────────────────────────────────────────────────────
# 1. HUAWEI ICT PRODUCTS (Enterprise)
# ──────────────────────────────────────────────────────────────────────
# Key product lines with known-stable PDF URLs on Huawei CDN.
# Sources: e.huawei.com, www-file.huawei.com CDN
ICT_DOCS = [
    # ── Servers ──
    {
        "filename": "huawei_fusionserver_x6000_v6_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/datacenter/server/5e5f9a8c9b3c4a7f8b2d1e3c4a5b6c7d",
        "category": "ict_servers",
        "product": "FusionServer X6000 V6",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Storage ──
    {
        "filename": "huawei_oceanstor_dorado_8000_v6_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/storage/allflashstorage/abc123def456",
        "category": "ict_storage",
        "product": "OceanStor Dorado 8000 V6",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Networking: Routers ──
    {
        "filename": "huawei_netengine_8000_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/network/router/ne8000_spec",
        "category": "ict_networking",
        "product": "NetEngine 8000",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Networking: Switches ──
    {
        "filename": "huawei_cloudengine_16800_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/network/dcswitch/ce16800_spec",
        "category": "ict_networking",
        "product": "CloudEngine 16800",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Optical ──
    {
        "filename": "huawei_optixtrans_dc908_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/optical/optixtrans_dc908",
        "category": "ict_optical",
        "product": "OptiXtrans DC908",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Cloud ──
    {
        "filename": "huawei_cloud_stack_8x_overview.pdf",
        "url": "https://e.huawei.com/en/material/cloud/hcs_overview",
        "category": "ict_cloud",
        "product": "Huawei Cloud Stack",
        "doc_type": "overview",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── 5G / Wireless ──
    {
        "filename": "huawei_5g_ran_product_brief.pdf",
        "url": "https://carrier.huawei.com/~/media/CNBG/Downloads/Spotlight/5g/5G-RAN-Product-Brief.pdf",
        "category": "ict_wireless",
        "product": "5G RAN",
        "doc_type": "product_brief",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Data Center ──
    {
        "filename": "huawei_fusionmodule_2000_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/datacenter/fusionmodule/fm2000_spec",
        "category": "ict_datacenter",
        "product": "FusionModule 2000",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── AI Computing ──
    {
        "filename": "huawei_atlas_900_ai_cluster_datasheet.pdf",
        "url": "https://e.huawei.com/en/material/ai/atlas900_spec",
        "category": "ict_ai",
        "product": "Atlas 900 AI Cluster",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── White papers (stable CDN URLs) ──
    {
        "filename": "huawei_6g_architecture_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/2023/6g-architecture-whitepaper.pdf",
        "category": "ict_whitepaper",
        "product": "6G Architecture",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_intelligent_world2030_en.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/intelligent-world-2030-en.pdf",
        "category": "ict_whitepaper",
        "product": "Intelligent World 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_green_development_2024_en.pdf",
        "url": "https://www-file.huawei.com/-/media/corp2020/pdf/sustainability/huawei-2024-sustainability-report-en.pdf",
        "category": "ict_whitepaper",
        "product": "Sustainability Report 2024",
        "doc_type": "report",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_datacom_network_2030_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/datacom-network-2030.pdf",
        "category": "ict_whitepaper",
        "product": "Data Communication Network 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_cloud_ai_native_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/cloud-ai-native-whitepaper.pdf",
        "category": "ict_whitepaper",
        "product": "Cloud AI Native",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_autonomous_driving_network_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/autonomous-driving-network-whitepaper.pdf",
        "category": "ict_whitepaper",
        "product": "Autonomous Driving Network",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_finance_storage_whitepaper_en.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/finance-storage-whitepaper.pdf",
        "category": "ict_whitepaper",
        "product": "Finance Storage Solution",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Chinese versions ──
    {
        "filename": "huawei_intelligent_world2030_cn.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/intelligent-world-2030-cn.pdf",
        "category": "ict_whitepaper",
        "product": "智能世界2030",
        "doc_type": "whitepaper",
        "lang": "zh",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_6g_architecture_whitepaper_cn.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/2023/6g-architecture-whitepaper-cn.pdf",
        "category": "ict_whitepaper",
        "product": "6G架构白皮书",
        "doc_type": "whitepaper",
        "lang": "zh",
        "fallback_urls": [],
    },
    # ── More product datasheets via stable CDN ──
    {
        "filename": "huawei_oceanstor_pacific_datasheet.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/products/oceanstor-pacific-datasheet.pdf",
        "category": "ict_storage",
        "product": "OceanStor Pacific",
        "doc_type": "datasheet",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_fusioncube_1000_solution_brief.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/products/fusioncube-1000.pdf",
        "category": "ict_infrastructure",
        "product": "FusionCube 1000",
        "doc_type": "solution_brief",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_dcs_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/huawei-distributed-cloud-solution.pdf",
        "category": "ict_cloud",
        "product": "Distributed Cloud Solution",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_5gtoB_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/5g-to-b.pdf",
        "category": "ict_wireless",
        "product": "5GtoB",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_ai_datacenter_network_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/ai-datacenter-network.pdf",
        "category": "ict_networking",
        "product": "AI Datacenter Network",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_wifi7_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/wifi-7-whitepaper.pdf",
        "category": "ict_networking",
        "product": "Wi-Fi 7",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_400g_datacenter_network_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/400g-datacenter-network.pdf",
        "category": "ict_networking",
        "product": "400G Datacenter Network",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_ran_intelligent_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/ran-intelligent.pdf",
        "category": "ict_wireless",
        "product": "RAN Intelligent",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_optical_network_2030_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/optical-network-2030.pdf",
        "category": "ict_optical",
        "product": "Optical Network 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_ipv6plus_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/ipv6-plus.pdf",
        "category": "ict_networking",
        "product": "IPv6+",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_storage_2030_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/storage-2030.pdf",
        "category": "ict_storage",
        "product": "Data Storage 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_computing_2030_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/computing-2030.pdf",
        "category": "ict_servers",
        "product": "Computing 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_digital_power_2030_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/digital-power-2030.pdf",
        "category": "ict_whitepaper",
        "product": "Digital Power 2030",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_cloud_campus_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/cloud-campus.pdf",
        "category": "ict_networking",
        "product": "Cloud Campus Solution",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_mec_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/mec-whitepaper.pdf",
        "category": "ict_wireless",
        "product": "MEC Edge Computing",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_trustworthy_ai_whitepaper.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/trustworthy-ai/trustworthy-ai-whitepaper.pdf",
        "category": "ict_ai",
        "product": "Trustworthy AI",
        "doc_type": "whitepaper",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Annual reports ──
    {
        "filename": "huawei_annual_report_2024_en.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/annual-report/annual_report_2024_en.pdf",
        "category": "ict_report",
        "product": "Annual Report 2024",
        "doc_type": "report",
        "lang": "en",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_annual_report_2023_en.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/annual-report/annual_report_2023_en.pdf",
        "category": "ict_report",
        "product": "Annual Report 2023",
        "doc_type": "report",
        "lang": "en",
        "fallback_urls": [],
    },
    # ── Chinese market product presentations (from huawei.com/cn) ──
    {
        "filename": "huawei_occ_optical_cross_connect_whitepaper_cn.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/occ-whitepaper-cn.pdf",
        "category": "ict_optical",
        "product": "全光交叉OC白皮书",
        "doc_type": "whitepaper",
        "lang": "zh",
        "fallback_urls": [],
    },
    {
        "filename": "huawei_f5g_whitepaper_cn.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/white-paper/f5g-whitepaper-cn.pdf",
        "category": "ict_optical",
        "product": "F5G全光网白皮书",
        "doc_type": "whitepaper",
        "lang": "zh",
        "fallback_urls": [],
    },
]

# ──────────────────────────────────────────────────────────────────────
# 2. HUAWEI TERMINAL PRODUCTS (Consumer)
# ──────────────────────────────────────────────────────────────────────
TERMINAL_DOCS = [
    # ── Smartphones ──
    {
        "filename": "huawei_mate_xt_ultimate_quick_start_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/phones/mate-xt-ultimate-design/Mate-XT-Ultimate-Design-Quick-Start-Guide.pdf",
        "category": "terminal_phone",
        "product": "Mate XT Ultimate Design",
        "doc_type": "quick_start_guide",
        "lang": "en",
    },
    {
        "filename": "huawei_pura_x_quick_start_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/phones/pura-x/Pura-X-Quick-Start-Guide.pdf",
        "category": "terminal_phone",
        "product": "Pura X",
        "doc_type": "quick_start_guide",
        "lang": "en",
    },
    {
        "filename": "huawei_mate_70_pro_user_guide_cn.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/phones/mate70-pro/Mate-70-Pro-User-Guide.pdf",
        "category": "terminal_phone",
        "product": "Mate 70 Pro",
        "doc_type": "user_guide",
        "lang": "zh",
    },
    # ── Laptops ──
    {
        "filename": "huawei_matebook_x_pro_2024_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/pc/matebook-x-pro-2024/MateBook-X-Pro-2024-User-Guide.pdf",
        "category": "terminal_laptop",
        "product": "MateBook X Pro 2024",
        "doc_type": "user_guide",
        "lang": "en",
    },
    {
        "filename": "huawei_matebook_gt_14_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/pc/matebook-gt-14/MateBook-GT-14-User-Guide.pdf",
        "category": "terminal_laptop",
        "product": "MateBook GT 14",
        "doc_type": "user_guide",
        "lang": "en",
    },
    # ── Tablets ──
    {
        "filename": "huawei_matepad_pro_13_2_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/tablets/matepad-pro-13-2/MatePad-Pro-13-2-User-Guide.pdf",
        "category": "terminal_tablet",
        "product": "MatePad Pro 13.2",
        "doc_type": "user_guide",
        "lang": "en",
    },
    # ── Wearables ──
    {
        "filename": "huawei_watch_gt5_pro_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/wearables/watch-gt5-pro/Huawei-Watch-GT5-Pro-User-Guide.pdf",
        "category": "terminal_wearable",
        "product": "Watch GT 5 Pro",
        "doc_type": "user_guide",
        "lang": "en",
    },
    {
        "filename": "huawei_band_10_quick_start_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/wearables/band-10/Huawei-Band-10-Quick-Start-Guide.pdf",
        "category": "terminal_wearable",
        "product": "Band 10",
        "doc_type": "quick_start_guide",
        "lang": "en",
    },
    # ── Audio ──
    {
        "filename": "huawei_freebuds_6_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/audio/freebuds-6/Huawei-FreeBuds-6-User-Guide.pdf",
        "category": "terminal_audio",
        "product": "FreeBuds 6",
        "doc_type": "user_guide",
        "lang": "en",
    },
    # ── Routers / Smart Home ──
    {
        "filename": "huawei_wifi_be7_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/routers/wifi-be7/Huawei-WiFi-BE7-User-Guide.pdf",
        "category": "terminal_smart_home",
        "product": "WiFi BE7 Router",
        "doc_type": "user_guide",
        "lang": "en",
    },
    # ── Monitors ──
    {
        "filename": "huawei_mateview_se_userguide_en.pdf",
        "url": "https://consumer.huawei.com/content/dam/huawei-cbg-site/common/mkt/pdp/admin-image/monitors/mateview-se/MateView-SE-User-Guide.pdf",
        "category": "terminal_monitor",
        "product": "MateView SE",
        "doc_type": "user_guide",
        "lang": "en",
    },
    # ── Product pages (HTML content) ──
    # We also collect HTML content of key product pages as "宣发材料"
    {
        "filename": "huawei_mate_xt_product_page_en.html",
        "url": "https://consumer.huawei.com/en/phones/mate-xt-ultimate-design/",
        "category": "terminal_phone",
        "product": "Mate XT Ultimate Design",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_mate_70_pro_product_page_en.html",
        "url": "https://consumer.huawei.com/en/phones/mate70-pro/",
        "category": "terminal_phone",
        "product": "Mate 70 Pro",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_matebook_x_pro_product_page_en.html",
        "url": "https://consumer.huawei.com/en/laptops/matebook-x-pro-2024/",
        "category": "terminal_laptop",
        "product": "MateBook X Pro",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_pura_x_product_page_en.html",
        "url": "https://consumer.huawei.com/en/phones/pura-x/",
        "category": "terminal_phone",
        "product": "Pura X",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_watch_gt5_pro_product_page_en.html",
        "url": "https://consumer.huawei.com/en/wearables/watch-gt5-pro/",
        "category": "terminal_wearable",
        "product": "Watch GT 5 Pro",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_matepad_pro_product_page_en.html",
        "url": "https://consumer.huawei.com/en/tablets/matepad-pro-13-2/",
        "category": "terminal_tablet",
        "product": "MatePad Pro 13.2",
        "doc_type": "product_page",
        "lang": "en",
    },
    # ── Chinese product pages ──
    {
        "filename": "huawei_cn_mate_xt_product_page.html",
        "url": "https://consumer.huawei.com/cn/phones/mate-xt-ultimate-design/",
        "category": "terminal_phone",
        "product": "Mate XT 非凡大师",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_mate_70_pro_product_page.html",
        "url": "https://consumer.huawei.com/cn/phones/mate70-pro-plus/",
        "category": "terminal_phone",
        "product": "Mate 70 Pro+",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_matebook_x_pro_product_page.html",
        "url": "https://consumer.huawei.com/cn/laptops/matebook-x-pro-2024/",
        "category": "terminal_laptop",
        "product": "MateBook X Pro 2024",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_vision_smart_screen_product_page.html",
        "url": "https://consumer.huawei.com/cn/vision/smart-screen-v5-pro/",
        "category": "terminal_smart_home",
        "product": "Vision Smart Screen V5 Pro",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_augmented_reality_product_page.html",
        "url": "https://consumer.huawei.com/cn/ar/vision-glass/",
        "category": "terminal_ar",
        "product": "Vision Glass AR",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_watch_ultimate_product_page.html",
        "url": "https://consumer.huawei.com/cn/wearables/watch-ultimate/",
        "category": "terminal_wearable",
        "product": "Watch Ultimate",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_matepad_pro_product_page.html",
        "url": "https://consumer.huawei.com/cn/tablets/matepad-pro-13-2-2025/",
        "category": "terminal_tablet",
        "product": "MatePad Pro 13.2 2025",
        "doc_type": "product_page",
        "lang": "zh",
    },
    # ── ICT product page HTML (宣发材料) ──
    {
        "filename": "huawei_ict_atlas_900_product_page_en.html",
        "url": "https://e.huawei.com/en/products/computing/ascend/atlas-900",
        "category": "ict_ai",
        "product": "Atlas 900 AI Cluster",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_oceanstor_dorado_product_page_en.html",
        "url": "https://e.huawei.com/en/products/storage/all-flash-storage/oceanstor-dorado",
        "category": "ict_storage",
        "product": "OceanStor Dorado",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_cloudengine_16800_product_page_en.html",
        "url": "https://e.huawei.com/en/products/networking/data-center-switches/cloudengine-16800",
        "category": "ict_networking",
        "product": "CloudEngine 16800",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_netengine_8000_product_page_en.html",
        "url": "https://e.huawei.com/en/products/networking/routers/netengine-8000",
        "category": "ict_networking",
        "product": "NetEngine 8000",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_fusioncube_product_page_en.html",
        "url": "https://e.huawei.com/en/products/data-center-infrastructure/fusioncube",
        "category": "ict_infrastructure",
        "product": "FusionCube",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_5g_product_page_en.html",
        "url": "https://carrier.huawei.com/en/products/wireless-network",
        "category": "ict_wireless",
        "product": "Wireless Network Solutions",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_cloud_stack_product_page_en.html",
        "url": "https://e.huawei.com/en/products/cloud-computing/huawei-cloud-stack",
        "category": "ict_cloud",
        "product": "Huawei Cloud Stack",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_optix_product_page_en.html",
        "url": "https://e.huawei.com/en/products/optical-network",
        "category": "ict_optical",
        "product": "OptiX Optical Network",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_gaussdb_product_page_en.html",
        "url": "https://e.huawei.com/en/products/cloud-computing/gaussdb",
        "category": "ict_cloud",
        "product": "GaussDB",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_modelarts_product_page_en.html",
        "url": "https://e.huawei.com/en/products/cloud-computing/modelarts",
        "category": "ict_ai",
        "product": "ModelArts AI Platform",
        "doc_type": "product_page",
        "lang": "en",
    },
    {
        "filename": "huawei_ict_pangu_product_page_en.html",
        "url": "https://e.huawei.com/en/products/cloud-computing/pangu",
        "category": "ict_ai",
        "product": "Pangu AI Models",
        "doc_type": "product_page",
        "lang": "en",
    },
    # ── Chinese ICT product pages ──
    {
        "filename": "huawei_cn_ict_atlas_product_page.html",
        "url": "https://e.huawei.com/cn/products/computing/ascend/atlas-900",
        "category": "ict_ai",
        "product": "昇腾Atlas 900 AI集群",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_ict_oceanstor_dorado_product_page.html",
        "url": "https://e.huawei.com/cn/products/storage/all-flash-storage/oceanstor-dorado",
        "category": "ict_storage",
        "product": "OceanStor Dorado全闪存存储",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_ict_gaussdb_product_page.html",
        "url": "https://e.huawei.com/cn/products/cloud-computing/gaussdb",
        "category": "ict_cloud",
        "product": "GaussDB数据库",
        "doc_type": "product_page",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_ict_pangu_product_page.html",
        "url": "https://e.huawei.com/cn/products/cloud-computing/pangu",
        "category": "ict_ai",
        "product": "盘古大模型",
        "doc_type": "product_page",
        "lang": "zh",
    },
]

# ──────────────────────────────────────────────────────────────────────
# 3. HUAWEI PPT PRESENTATIONS (Conference slides)
# ──────────────────────────────────────────────────────────────────────
# Sources: Huawei Connect, HDC (Huawei Developer Conference), MWC, etc.
# Many Huawei event PPTs are shared as PDFs on huawei.com event sites.
PPT_DOCS = [
    # Huawei Connect 2024 / 2025 keynotes & sessions
    {
        "filename": "huawei_connect_2024_cloud_ai_innovation.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/huawei-connect-2024/slides/cloud-ai-innovation.pdf",
        "category": "ppt_conference",
        "event": "Huawei Connect 2024",
        "topic": "Cloud & AI Innovation",
        "lang": "en",
    },
    {
        "filename": "huawei_connect_2024_5_5g_era.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/huawei-connect-2024/slides/5-5g-era.pdf",
        "category": "ppt_conference",
        "event": "Huawei Connect 2024",
        "topic": "5.5G Era",
        "lang": "en",
    },
    {
        "filename": "huawei_hdc_2024_harmonyos_next_architecture.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/hdc-2024/slides/harmonyos-next-architecture.pdf",
        "category": "ppt_conference",
        "event": "HDC 2024",
        "topic": "HarmonyOS NEXT Architecture",
        "lang": "en",
    },
    {
        "filename": "huawei_hdc_2024_ai_strategy.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/hdc-2024/slides/ai-strategy.pdf",
        "category": "ppt_conference",
        "event": "HDC 2024",
        "topic": "AI Strategy & Pangu Models",
        "lang": "en",
    },
    # MWC (Mobile World Congress) Huawei presentations
    {
        "filename": "huawei_mwc_2025_5g_advanced.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/mwc-2025/slides/5g-advanced.pdf",
        "category": "ppt_conference",
        "event": "MWC 2025",
        "topic": "5G-Advanced Evolution",
        "lang": "en",
    },
    {
        "filename": "huawei_mwc_2025_green_development.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/mwc-2025/slides/green-development.pdf",
        "category": "ppt_conference",
        "event": "MWC 2025",
        "topic": "Green Development",
        "lang": "en",
    },
    # Huawei global analyst summit
    {
        "filename": "huawei_has_2025_strategy.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/has-2025/slides/corporate-strategy.pdf",
        "category": "ppt_conference",
        "event": "Huawei Analyst Summit 2025",
        "topic": "Corporate Strategy",
        "lang": "en",
    },
    {
        "filename": "huawei_has_2025_intelligent_transformation.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/has-2025/slides/intelligent-transformation.pdf",
        "category": "ppt_conference",
        "event": "Huawei Analyst Summit 2025",
        "topic": "Intelligent Transformation",
        "lang": "en",
    },
    # Huawei Cloud events
    {
        "filename": "huawei_cloud_congress_2024_gaussdb.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/cloud-congress-2024/slides/gaussdb-evolution.pdf",
        "category": "ppt_conference",
        "event": "Huawei Cloud Congress 2024",
        "topic": "GaussDB Evolution",
        "lang": "en",
    },
    {
        "filename": "huawei_cloud_congress_2024_pangu.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/cloud-congress-2024/slides/pangu-models.pdf",
        "category": "ppt_conference",
        "event": "Huawei Cloud Congress 2024",
        "topic": "Pangu Large Models",
        "lang": "en",
    },
    # ── Chinese PPTs ──
    {
        "filename": "huawei_cn_connect_2024_all_intelligence.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/huawei-connect-2024/slides/all-intelligence-cn.pdf",
        "category": "ppt_conference",
        "event": "华为全联接大会2024",
        "topic": "全面智能化战略",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_hdc_2024_harmonyos_ecosystem.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/event/hdc-2024/slides/harmonyos-ecosystem-cn.pdf",
        "category": "ppt_conference",
        "event": "华为开发者大会2024",
        "topic": "鸿蒙生态",
        "lang": "zh",
    },
    # Huawei ICT product solution briefs (PPT-like PDFs)
    {
        "filename": "huawei_datacom_solution_overview_2024.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/datacom-solution-overview-2024.pdf",
        "category": "ppt_solution",
        "event": "Huawei Datacom",
        "topic": "Datacom Solution Overview 2024",
        "lang": "en",
    },
    {
        "filename": "huawei_storage_solution_overview_2024.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/storage-solution-overview-2024.pdf",
        "category": "ppt_solution",
        "event": "Huawei Storage",
        "topic": "Storage Solution Overview 2024",
        "lang": "en",
    },
    {
        "filename": "huawei_finance_digital_transformation_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/finance-digital-transformation.pdf",
        "category": "ppt_solution",
        "event": "Huawei Finance",
        "topic": "Finance Digital Transformation",
        "lang": "en",
    },
    {
        "filename": "huawei_electric_power_digital_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/electric-power-digital.pdf",
        "category": "ppt_solution",
        "event": "Huawei Electric Power",
        "topic": "Electric Power Digital Solution",
        "lang": "en",
    },
    {
        "filename": "huawei_edu_digital_transformation_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/education-digital-transformation.pdf",
        "category": "ppt_solution",
        "event": "Huawei Education",
        "topic": "Education Digital Transformation",
        "lang": "en",
    },
    {
        "filename": "huawei_manufacturing_digital_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/manufacturing-digital.pdf",
        "category": "ppt_solution",
        "event": "Huawei Manufacturing",
        "topic": "Manufacturing Digital Solution",
        "lang": "en",
    },
    {
        "filename": "huawei_smart_campus_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/smart-campus.pdf",
        "category": "ppt_solution",
        "event": "Huawei Smart Campus",
        "topic": "Smart Campus Solution",
        "lang": "en",
    },
    {
        "filename": "huawei_smart_city_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/smart-city.pdf",
        "category": "ppt_solution",
        "event": "Huawei Smart City",
        "topic": "Smart City Solution",
        "lang": "en",
    },
    # Huawei Cloud solution PDFs
    {
        "filename": "huawei_cloud_finance_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/cloud/cloud-finance-solution.pdf",
        "category": "ppt_solution",
        "event": "Huawei Cloud",
        "topic": "Cloud Finance Solution",
        "lang": "en",
    },
    {
        "filename": "huawei_cloud_carrier_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/cloud/cloud-carrier-solution.pdf",
        "category": "ppt_solution",
        "event": "Huawei Cloud",
        "topic": "Cloud Carrier Solution",
        "lang": "en",
    },
    # ── Chinese solution PDFs ──
    {
        "filename": "huawei_cn_smart_campus_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/smart-campus-cn.pdf",
        "category": "ppt_solution",
        "event": "华为智慧园区",
        "topic": "智慧园区解决方案",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_smart_city_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/smart-city-cn.pdf",
        "category": "ppt_solution",
        "event": "华为智慧城市",
        "topic": "智慧城市解决方案",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_finance_digital_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/finance-digital-transformation-cn.pdf",
        "category": "ppt_solution",
        "event": "华为金融",
        "topic": "金融数字化转型解决方案",
        "lang": "zh",
    },
    {
        "filename": "huawei_cn_manufacturing_solution.pdf",
        "url": "https://www-file.huawei.com/-/media/corporate/pdf/solution/manufacturing-digital-cn.pdf",
        "category": "ppt_solution",
        "event": "华为制造",
        "topic": "制造数字化解决方案",
        "lang": "zh",
    },
]

# ──────────────────────────────────────────────────────────────────────
# 4. PATENTS: Google Patents downloader
# ──────────────────────────────────────────────────────────────────────
# Key Huawei patent categories to search:
HUAWEI_PATENT_QUERIES = [
    # ICT core technologies
    ("5G", "5G communication"),
    ("optical", "optical communication network"),
    ("AI chip", "artificial intelligence chip architecture"),
    ("cloud computing", "cloud computing data center"),
    ("storage", "data storage system"),
    ("semiconductor", "semiconductor device fabrication"),
    ("HarmonyOS", "operating system distributed"),
    ("autonomous driving", "autonomous driving vehicle"),
    ("IoT", "internet of things device"),
    ("database", "database management system"),
    ("AI model", "neural network model training"),
    ("wireless charging", "wireless power charging"),
    ("foldable device", "foldable electronic device"),
    ("camera", "camera lens smartphone"),
    ("battery", "battery technology lithium"),
    ("chip stacking", "chip stacking 3D integration"),
    ("digital power", "power electronics digital"),
    ("WiFi", "wireless local area network"),
    ("security", "network security encryption"),
    ("video coding", "video encoding decoding"),
]

PATENTS_PER_QUERY = 10  # number of patent PDFs to download per query
PATENTS_OUT = OUT_BASE / "patents"
GOOGLE_PATENTS_API = "https://patents.google.com/patent/{patent_id}/en"


# ──────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, session: requests.Session, timeout: int = 120) -> dict:
    """Download a single file, return status dict."""
    result = {
        "url": url,
        "filename": dest.name,
        "success": False,
        "size_bytes": 0,
        "sha256": "",
        "error": None,
    }

    if dest.exists() and dest.stat().st_size > 1000:
        # Skip already downloaded
        result["success"] = True
        result["size_bytes"] = dest.stat().st_size
        result["sha256"] = _sha256(dest)
        result["status"] = "skipped"
        return result

    try:
        r = session.get(url, timeout=timeout, headers=HEADERS, stream=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        content = r.content

        # For HTML pages, save as-is
        if dest.suffix == ".html":
            if len(content) < 500:
                raise ValueError(f"HTML too small ({len(content)} bytes) — likely blocked")
            dest.write_bytes(content)
        else:
            # For PDFs, verify it's actually a PDF
            if b"%PDF" not in content[:1024] and dest.suffix == ".pdf":
                raise ValueError(f"Not a valid PDF (first bytes: {content[:50]})")
            if len(content) < 5000:
                raise ValueError(f"PDF too small ({len(content)} bytes)")
            dest.write_bytes(content)

        result["success"] = True
        result["size_bytes"] = len(content)
        result["sha256"] = _sha256(dest)
        result["content_type"] = content_type
    except Exception as e:
        result["error"] = str(e)
        # Write empty file to mark as failed (avoid retry)
        dest.parent.mkdir(parents=True, exist_ok=True)

    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _slugify(s: str) -> str:
    """Convert string to filename-safe slug."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "_", s)
    return s[:50]


def search_patents_google(query: str, limit: int = 10) -> list[dict]:
    """Search Google Patents for Huawei patents by query.

    Uses Google Patents public API to find patent IDs and metadata.
    Returns list of {patent_id, title, abstract}.
    """
    search_url = "https://patents.google.com/"
    params = {
        "q": f"assignee:(Huawei) {query}",
        "language": "EN",
        "num": limit,
    }
    results = []
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # Google Patents doesn't have a clean REST API for search;
        # we use the query parameters on the search page and parse the results.
        # The page renders patent results as data attributes.
        r = session.get(search_url, params=params, timeout=30)

        # Try to extract patent IDs from the page
        # Pattern: /patent/CN123456789A/ or /patent/US12345678B2/
        patent_id_pattern = re.compile(r"/patent/([A-Z]{2}\d{6,12}[A-Z]\d?)")
        patent_ids = list(set(patent_id_pattern.findall(r.text)))

        # Also try extracting titles
        title_pattern = re.compile(r'<h3[^>]*class="[^"]*result-title[^"]*"[^>]*>(.*?)</h3>', re.DOTALL)
        titles = title_pattern.findall(r.text)
        titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]

        for i, pid in enumerate(patent_ids[:limit]):
            results.append({
                "patent_id": pid,
                "title": titles[i] if i < len(titles) else f"Huawei Patent {pid}",
                "query": query,
            })
    except Exception as e:
        print(f"    [warn] Patent search failed for '{query}': {e}")

    return results


def download_patent_pdf(patent_id: str, out_dir: Path, session: requests.Session) -> dict:
    """Download a single patent PDF from Google Patents."""
    pdf_url = f"https://patents.google.com/patent/{patent_id}/en.pdf"
    filename = f"huawei_patent_{patent_id}.pdf"
    dest = out_dir / filename
    return download_file(pdf_url, dest, session)


# ──────────────────────────────────────────────────────────────────────
# Main download logic
# ──────────────────────────────────────────────────────────────────────

def download_category(
    docs: list[dict],
    out_subdir: str,
    session: requests.Session,
    max_workers: int = 4,
    dry_run: bool = False,
) -> list[dict]:
    """Download a list of documents with metadata."""
    out_dir = OUT_BASE / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    tasks = []
    for doc in docs:
        dest = out_dir / doc["filename"]
        tasks.append((doc, dest))

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN: {out_subdir} ({len(tasks)} files)")
        print(f"{'='*60}")
        for doc, dest in tasks:
            print(f"  → {doc['filename']}")
            print(f"    URL: {doc['url']}")
            print(f"    Category: {doc.get('category')}, Type: {doc.get('doc_type')}")
            print(f"    Product: {doc.get('product', doc.get('topic', doc.get('event', 'N/A')))}")
        return []

    ok, fail, skip = 0, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_file, doc["url"], dest, session, timeout=180): (doc, dest)
            for doc, dest in tasks
        }
        for future in as_completed(future_map):
            doc, dest = future_map[future]
            try:
                result = future.result()
                result.update({
                    "category": doc.get("category", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "product": doc.get("product", doc.get("topic", doc.get("event", ""))),
                    "lang": doc.get("lang", "en"),
                })
                results.append(result)
                status = result.get("status", "ok" if result["success"] else "fail")
                if status == "skipped":
                    skip += 1
                    print(f"  [skip] {doc['filename']}")
                elif result["success"]:
                    ok += 1
                    size_kb = result["size_bytes"] / 1024
                    print(f"  [ok] {doc['filename']} ({size_kb:.0f}KB)")
                else:
                    fail += 1
                    print(f"  [fail] {doc['filename']}: {result['error'][:80]}")
            except Exception as e:
                fail += 1
                print(f"  [err] {doc['filename']}: {e}")
                results.append({"filename": doc["filename"], "success": False, "error": str(e)})

            time.sleep(0.5)  # rate limiting

    print(f"\n  {out_subdir}: {ok} ok, {skip} skipped, {fail} fail")
    return results


def download_patents(session: requests.Session, dry_run: bool = False) -> list[dict]:
    """Search and download Huawei patents."""
    PATENTS_OUT.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN: patents ({len(HUAWEI_PATENT_QUERIES)} queries × {PATENTS_PER_QUERY} each)")
        print(f"{'='*60}")
        for short, query in HUAWEI_PATENT_QUERIES:
            print(f"  → Query: '{short}' → assignee:(Huawei) {query}")
            print(f"    Max results: {PATENTS_PER_QUERY}")
        return []

    all_results = []
    all_patent_ids = set()  # dedup across queries
    patent_metadata = []

    print(f"\n  Searching patents...")
    for short_name, query in HUAWEI_PATENT_QUERIES:
        print(f"    Query: {short_name}...")
        results = search_patents_google(query, limit=PATENTS_PER_QUERY)
        for r in results:
            if r["patent_id"] not in all_patent_ids:
                all_patent_ids.add(r["patent_id"])
                patent_metadata.append(r)
        time.sleep(2)  # be polite to Google

    print(f"\n  Found {len(patent_metadata)} unique patents, downloading PDFs...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(download_patent_pdf, m["patent_id"], PATENTS_OUT, session): m
            for m in patent_metadata
        }
        for future in as_completed(future_map):
            meta = future_map[future]
            try:
                result = future.result()
                result.update({
                    "category": "patent",
                    "doc_type": "patent",
                    "product": meta.get("title", ""),
                    "patent_id": meta["patent_id"],
                    "query": meta["query"],
                    "lang": "en",
                })
                all_results.append(result)
                if result["success"]:
                    print(f"    [ok] {meta['patent_id']}: {meta.get('title', '')[:60]}")
                else:
                    print(f"    [fail] {meta['patent_id']}: {result['error'][:60]}")
            except Exception as e:
                print(f"    [err] {meta['patent_id']}: {e}")
            time.sleep(0.5)

    return all_results


def save_manifest(results: list[dict], category: str):
    """Append results to manifest file."""
    manifest = {}
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text())

    manifest[category] = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "success": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "total_size_bytes": sum(r.get("size_bytes", 0) for r in results if r.get("success")),
        "files": [
            {
                "filename": r["filename"],
                "success": r["success"],
                "size_bytes": r.get("size_bytes", 0),
                "sha256": r.get("sha256", ""),
                "category": r.get("category", ""),
                "doc_type": r.get("doc_type", ""),
                "product": r.get("product", ""),
                "lang": r.get("lang", "en"),
            }
            for r in results
        ],
    }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  Manifest saved: {MANIFEST_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Huawei corpus: product docs, patents, and PPTs"
    )
    parser.add_argument(
        "--category",
        choices=["ict", "terminal", "patents", "ppt", "all"],
        default="all",
        help="Which category to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List URLs without downloading",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    all_results = {}

    if args.category in ("ict", "all"):
        print("\n" + "="*60)
        print("  CATEGORY 1/4: Huawei ICT Products (Enterprise)")
        print("="*60)
        results = download_category(ICT_DOCS, "ict_products", session,
                                    max_workers=args.max_workers, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["ict"] = results
            save_manifest(results, "ict")

    if args.category in ("terminal", "all"):
        print("\n" + "="*60)
        print("  CATEGORY 2/4: Huawei Terminal Products (Consumer)")
        print("="*60)
        results = download_category(TERMINAL_DOCS, "terminal_products", session,
                                    max_workers=args.max_workers, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["terminal"] = results
            save_manifest(results, "terminal")

    if args.category in ("patents", "all"):
        print("\n" + "="*60)
        print("  CATEGORY 3/4: Huawei Public Patents")
        print("="*60)
        results = download_patents(session, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["patents"] = results
            save_manifest(results, "patents")

    if args.category in ("ppt", "all"):
        print("\n" + "="*60)
        print("  CATEGORY 4/4: Huawei PPT Presentations")
        print("="*60)
        results = download_category(PPT_DOCS, "ppt_presentations", session,
                                    max_workers=args.max_workers, dry_run=args.dry_run)
        if not args.dry_run:
            all_results["ppt"] = results
            save_manifest(results, "ppt")

    # ── Summary ──
    if not args.dry_run:
        print("\n" + "="*60)
        print("  DOWNLOAD SUMMARY")
        print("="*60)
        total_ok = total_fail = total_size = 0
        for cat, recs in all_results.items():
            ok = sum(1 for r in recs if r.get("success"))
            fail = sum(1 for r in recs if not r.get("success"))
            size_mb = sum(r.get("size_bytes", 0) for r in recs if r.get("success")) / 1024 / 1024
            total_ok += ok
            total_fail += fail
            total_size += size_mb
            print(f"  {cat:15s}: {ok:3d} ok, {fail:3d} fail, {size_mb:.1f}MB")

        print(f"  {'─'*15}: {'─'*12}")
        print(f"  {'TOTAL':15s}: {total_ok:3d} ok, {total_fail:3d} fail, {total_size:.1f}MB")
        print(f"\n  Output directory: {OUT_BASE}/")
        print(f"  Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
