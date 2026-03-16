#!/usr/bin/env python3
"""
Minimal API connectivity test.
No git clone needed — copy-paste this single file to server and run.

Setup & run (3 commands):
    python3 -m venv /tmp/api_test && source /tmp/api_test/bin/activate
    pip install requests
    COMPANY_API_KEY="sk-your-key" python3 test_api_connectivity.py
"""
import os, json, sys, time

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ============ CONFIG — 只改这里 ============
API_KEY = os.environ.get("COMPANY_API_KEY", "sk-PUT-YOUR-KEY-HERE")
API_URL = os.environ.get("COMPANY_API_URL", "https://az.gptplus5.com")
MODEL   = "claude-sonnet-4-20250514"
# ===========================================

ENDPOINT = f"{API_URL.rstrip('/')}/v1/chat/completions"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
PAYLOAD = {"model": MODEL, "max_tokens": 20,
           "messages": [{"role": "user", "content": "Say hello in 3 words."}]}


def test(label, proxies=None, timeout=30):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"  URL   : {ENDPOINT}")
    print(f"  Proxy : {proxies if proxies else 'NONE (direct)'}")
    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=HEADERS, json=PAYLOAD,
                          proxies=proxies, timeout=timeout, verify=False)
        dt = time.time() - t0
        print(f"  Status: {r.status_code}  ({dt:.1f}s)")
        try:
            print(f"  Body  : {json.dumps(r.json(), ensure_ascii=False)[:500]}")
        except Exception:
            print(f"  Body  : {r.text[:500]}")
    except Exception as e:
        print(f"  ERROR : {type(e).__name__}: {e}")


if __name__ == "__main__":
    if API_KEY == "sk-PUT-YOUR-KEY-HERE":
        print("WARNING: API_KEY not set! Run with: COMPANY_API_KEY=sk-xxx python3 test_api_connectivity.py")
    print(f"API_KEY : {'***' + API_KEY[-6:] if len(API_KEY) > 10 else '(NOT SET)'}")
    print(f"MODEL   : {MODEL}")
    print(f"ENV proxy: HTTP_PROXY={os.environ.get('HTTP_PROXY','')}")
    print(f"ENV proxy: HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')}")

    # 1) Direct — bypass any env proxy
    test("1. Direct (no proxy)", proxies={"http": "", "https": ""})

    # 2) Use env proxy if set
    env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_proxy:
        test(f"2. Env proxy ({env_proxy})", proxies=None)
    else:
        print("\n[SKIP] Test 2: no HTTP(S)_PROXY in env")

    # 3) Mentor's proxy — uncomment and fill in IP
    # test("3. Mentor proxy", proxies={"http": "http://10.194.x.x:3128",
    #                                   "https": "http://10.194.x.x:3128"})

    print(f"\n{'='*50}")
    print("DONE.  200=OK | 401=bad key | 429=no quota | timeout=blocked")
