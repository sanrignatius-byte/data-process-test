"""Quick probe: test company API response format."""
import requests, json, os

url = os.environ.get("COMPANY_API_URL", "")
if not url:
    print("ERROR: COMPANY_API_URL not set. Copy .env.example to .env and fill in values.")
    exit(1)
key = os.environ.get("COMPANY_API_KEY", "")

print(f"URL: {url}")
print(f"Key: {key[:8]}...{key[-4:]}" if len(key) > 12 else f"Key: {key}")

# --- Test 1: non-streaming ---
print("\n=== Test 1: non-streaming ===")
r = requests.post(
    url,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    json={
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50,
        "stream": False,
    },
    verify=False,
    timeout=30,
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:1000]}")

# --- Test 2: streaming ---
print("\n=== Test 2: streaming ===")
r2 = requests.post(
    url,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    json={
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 50,
        "stream": True,
    },
    verify=False,
    timeout=30,
    stream=True,
)
print(f"Status: {r2.status_code}")
line_count = 0
for line in r2.iter_lines():
    line_count += 1
    if line_count <= 10:
        print(f"  Line {line_count}: {line!r}")
print(f"Total lines: {line_count}")
