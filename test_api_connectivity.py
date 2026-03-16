#!/usr/bin/env python3
"""
Minimal API connectivity test — ZERO dependencies, stdlib only.
Copy-paste to server and run:
    COMPANY_API_KEY="sk-your-key" python3 test_api_connectivity.py
"""
import os, json, ssl, time, urllib.request, urllib.error

# ============ CONFIG — 只改这里 ============
API_KEY = os.environ.get("COMPANY_API_KEY", "sk-PUT-YOUR-KEY-HERE")
API_URL = os.environ.get("COMPANY_API_URL", "https://az.gptplus5.com")
MODEL   = "claude-sonnet-4-20250514"
# ===========================================

ENDPOINT = f"{API_URL.rstrip('/')}/v1/chat/completions"
PAYLOAD  = json.dumps({"model": MODEL, "max_tokens": 20,
                        "messages": [{"role": "user", "content": "Say hi in 3 words."}]}).encode()

# Skip SSL verify (corporate proxies often have custom certs)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def test(label, proxy_url=None, timeout=30):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"  URL   : {ENDPOINT}")
    print(f"  Proxy : {proxy_url or 'NONE (direct)'}")

    req = urllib.request.Request(
        ENDPOINT, data=PAYLOAD, method="POST",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})

    if proxy_url:
        handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    else:
        handler = urllib.request.ProxyHandler({})  # no proxy

    opener = urllib.request.build_opener(handler, urllib.request.HTTPSHandler(context=CTX))

    try:
        t0 = time.time()
        resp = opener.open(req, timeout=timeout)
        dt = time.time() - t0
        body = resp.read().decode()
        print(f"  Status: {resp.status}  ({dt:.1f}s)")
        print(f"  Body  : {body[:500]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  Status: {e.code}  {e.reason}")
        print(f"  Body  : {body[:500]}")
    except Exception as e:
        print(f"  ERROR : {type(e).__name__}: {e}")


if __name__ == "__main__":
    if API_KEY == "sk-PUT-YOUR-KEY-HERE":
        print("WARNING: set key first!  COMPANY_API_KEY=sk-xxx python3 test_api_connectivity.py")
    print(f"API_KEY : {'***' + API_KEY[-6:] if len(API_KEY) > 10 else '(NOT SET)'}")
    print(f"MODEL   : {MODEL}")
    print(f"ENV     : HTTP_PROXY={os.environ.get('HTTP_PROXY','')}")
    print(f"ENV     : HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')}")

    # Test 1: Direct (bypass env proxy)
    test("1. Direct (no proxy)")

    # Test 2: Use env proxy if set
    env_px = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_px:
        test(f"2. Env proxy ({env_px})", proxy_url=env_px)
    else:
        print("\n[SKIP] Test 2: no HTTP(S)_PROXY in env")

    # Test 3: Mentor proxy — uncomment and fill IP
    # test("3. Mentor proxy", proxy_url="http://10.194.x.x:3128")

    print(f"\n{'='*50}")
    print("DONE.  200=OK | 401=bad key | 429=no quota | timeout=blocked")
