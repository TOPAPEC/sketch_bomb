import os
import json
import base64
import io
import requests
from pathlib import Path
from PIL import Image

def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.5"

def img_to_b64(path):
    img = Image.open(path).resize((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

test_imgs = list(Path("out3/v7_experiments/").glob("exp1_baseline_cat*.png"))[:2]
if not test_imgs:
    test_imgs = list(Path("out3/v7_experiments/").glob("exp1_*.png"))[:2]

print(f"Testing with {len(test_imgs)} images: {[p.name for p in test_imgs]}")

content = [{"type": "text", "text": (
    "You see 2 candidate images of 'cat'. "
    "Pick the best one by visual quality and coherence. "
    "Respond ONLY with JSON: {\"reasoning\": \"...\", \"best_image\": N} where N is 1 or 2."
)}]

for i, p in enumerate(test_imgs):
    b64 = img_to_b64(p)
    content.append({"type": "text", "text": f"Image {i+1}:"})
    content.append({"type": "image_url",
                   "image_url": {"url": f"data:image/png;base64,{b64}"}})

print("\n--- Test 1: with response_format json_object ---")
try:
    r = requests.post(URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL,
              "messages": [{"role": "user", "content": content}],
              "response_format": {"type": "json_object"},
              "max_tokens": 512, "temperature": 0.3},
        timeout=90)
    print(f"Status: {r.status_code}")
    data = r.json()
    if "error" in data:
        print(f"API Error: {data['error']}")
    elif "choices" in data:
        raw = data["choices"][0]["message"]["content"]
        print(f"Raw response: {repr(raw[:500])}")
        try:
            parsed = json.loads(raw)
            print(f"Parsed: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
    else:
        print(f"Unexpected response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"Exception: {e}")

print("\n--- Test 2: without response_format, prompt-based JSON ---")
content2 = [{"type": "text", "text": (
    "You see 2 candidate images of 'cat'. "
    "Pick the best one by visual quality. "
    "You MUST respond with ONLY a JSON object, no other text: "
    "{\"reasoning\": \"brief reason\", \"best_image\": 1}"
)}]
for i, p in enumerate(test_imgs):
    b64 = img_to_b64(p)
    content2.append({"type": "text", "text": f"Image {i+1}:"})
    content2.append({"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}})

try:
    r = requests.post(URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL,
              "messages": [{"role": "user", "content": content2}],
              "max_tokens": 256, "temperature": 0.1},
        timeout=90)
    print(f"Status: {r.status_code}")
    data = r.json()
    if "error" in data:
        print(f"API Error: {data['error']}")
    elif "choices" in data:
        raw = data["choices"][0]["message"]["content"]
        print(f"Raw response: {repr(raw[:500])}")
        try:
            parsed = json.loads(raw)
            print(f"Parsed: {json.dumps(parsed, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            idx_match = None
            for token in raw.split():
                if token.strip('{},"').isdigit():
                    idx_match = token.strip('{},"')
                    break
            if idx_match:
                print(f"Extracted number: {idx_match}")
    else:
        print(f"Unexpected: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"Exception: {e}")

print("\n--- Test 3: simple text-only test ---")
try:
    r = requests.post(URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL,
              "messages": [{"role": "user", "content": "Reply with only: {\"test\": 1}"}],
              "response_format": {"type": "json_object"},
              "max_tokens": 64, "temperature": 0},
        timeout=30)
    data = r.json()
    if "error" in data:
        print(f"API Error: {data['error']}")
    elif "choices" in data:
        raw = data["choices"][0]["message"]["content"]
        print(f"Raw: {repr(raw)}")
except Exception as e:
    print(f"Exception: {e}")
