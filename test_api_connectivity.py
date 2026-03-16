"""
Minimal API connectivity test — zero external dependencies beyond `requests`.
Usage:
    pip install requests
    python test_api_connectivity.py
"""
import os, json, time, requests

# ============ CONFIG — 只改这里 ============
API_KEY = os.environ.get("COMPANY_API_KEY", "sk-PUT-YOUR-KEY-HERE")
API_URL = os.environ.get("COMPANY_API_URL", "https://az.gptplus5.com")
MODEL   = "claude-sonnet-4-20250514"
PROXY   = os.environ.get("HTTPS_PROXY", "") or os.environ.get("HTTP_PROXY", "")
# ===========================================

ENDPOINT = f"{API_URL.rstrip('/')}/v1/chat/completions"

def test(label: str, proxies: dict | None, timeout: int = 30):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"  URL   : {ENDPOINT}")
    print(f"  Proxy : {proxies or 'NONE (direct)'}")
    print(f"{'='*50}")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Say hello in 3 words."}],
    }
    try:
        t0 = time.time()
        r = requests.post(ENDPOINT, headers=headers, json=payload,
                          proxies=proxies, timeout=timeout)
        elapsed = time.time() - t0
        print(f"  Status: {r.status_code}  ({elapsed:.1f}s)")
        try:
            body = r.json()
            print(f"  Body  : {json.dumps(body, ensure_ascii=False)[:500]}")
        except Exception:
            print(f"  Body  : {r.text[:500]}")
        return r.status_code
    except Exception as e:
        print(f"  ERROR : {type(e).__name__}: {e}")
        return None

if __name__ == "__main__":
    print(f"API_KEY : {'***' + API_KEY[-6:] if len(API_KEY)>10 else '(not set)'}")
    print(f"MODEL   : {MODEL}")

    # Test 1: No proxy (direct)
    test("Direct (no proxy)", proxies={"http": "", "https": ""})

    # Test 2: System proxy (if set)
    if PROXY:
        test(f"System proxy ({PROXY})", proxies=None)  # None = use env
    else:
        print("\n[SKIP] No system proxy set in env.")

    # Test 3: Explicit proxy — add more as needed
    EXTRA_PROXIES = [
        # ("10.194.x.x:3128", {"http": "http://10.194.x.x:3128", "https": "http://10.194.x.x:3128"}),
    ]
    for name, px in EXTRA_PROXIES:
        test(f"Explicit proxy {name}", proxies=px)

    print("\n" + "="*50)
    print("DONE. Check status codes above.")
    print("  200 = OK | 401 = bad key | 429 = quota | timeout = network blocked")
