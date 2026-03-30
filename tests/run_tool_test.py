"""
End-to-end test: start ORT server and call the tool-calling endpoint.
Usage:
    conda run -n ofoundrydev python tests/run_tool_test.py
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://127.0.0.1:18000"
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "phi3-mini-int4")

TOOL_REQUEST = {
    "model": "phi3-mini-int4",
    "messages": [
        {"role": "user", "content": "What is the weather like in Tokyo?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as r:
        return json.loads(r.read())


def _post(path: str, body: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    # --- Try to reach an already-running server first ---
    already_up = False
    try:
        health = _get("/health")
        print(f"[INFO] Server already running: {health}")
        already_up = True
    except Exception:
        pass

    proc = None
    if not already_up:
        print(f"[INFO] Starting ORT server with model {MODEL_PATH} ...")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "ofoundry.cli",
                "serve", "--engine", "ort", "-m", MODEL_PATH,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait for server to be ready
        for attempt in range(20):
            time.sleep(2)
            try:
                health = _get("/health")
                print(f"[INFO] Server ready: {health}")
                break
            except Exception:
                print(f"[INFO] Waiting for server... ({attempt + 1}/20)")
        else:
            if proc:
                proc.terminate()
                out, _ = proc.communicate(timeout=5)
                print("[ERROR] Server failed to start. Output:")
                print(out.decode(errors="replace"))
            sys.exit(1)

    # --- Unit tests (no server needed) ---
    print("\n=== Unit tests ===")
    import subprocess as sp
    r = sp.run(
        [sys.executable, "-m", "pytest", "tests/test_tools.py", "-v"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    print(r.stdout)
    if r.returncode != 0:
        print("[FAIL] Unit tests failed")
        print(r.stderr)
    else:
        print("[PASS] All unit tests passed")

    # --- Tool calling request ---
    print("\n=== Tool calling (end-to-end) ===")
    print(f"[INFO] Sending request (may take up to 120s for phi3-mini)...")
    try:
        resp = _post("/v1/chat/completions", TOOL_REQUEST, timeout=120)
    except urllib.error.URLError as e:
        print(f"[ERROR] Request failed: {e}")
        if proc:
            proc.terminate()
        sys.exit(1)

    print("[INFO] Response:")
    print(json.dumps(resp, indent=2, ensure_ascii=False))

    choices = resp.get("choices", [])
    if not choices:
        print("[FAIL] No choices in response")
    else:
        msg = choices[0].get("message", {})
        finish = choices[0].get("finish_reason", "")
        if finish == "tool_calls" and msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            print(f"\n[PASS] Tool call detected!")
            print(f"       name      : {tc['function']['name']}")
            print(f"       arguments : {tc['function']['arguments']}")
        elif msg.get("content"):
            print(f"\n[WARN] Model responded with text instead of tool call:")
            print(f"       {msg['content'][:200]}")
            print("       (This is expected for small models that may not follow instructions)")
        else:
            print(f"\n[FAIL] Unexpected response format")

    if proc:
        proc.terminate()


if __name__ == "__main__":
    main()
