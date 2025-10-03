# utils/ollama_client.py
import os
import requests
import json
from typing import Optional, Dict, Any

# Default values from environment
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def call_ollama_generate(
    prompt: str,
    model: str = "llama3",
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 60,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Call Ollama's /api/generate endpoint with full support for custom base_url and api_key.
    Returns the generated text as a string.
    """
    base = base_url or DEFAULT_OLLAMA_BASE
    key = api_key if api_key is not None else DEFAULT_OLLAMA_API_KEY
    url = f"{base.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = _build_headers(key)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()

        # Try to parse streaming-like JSON lines or plain JSON
        text_out = ""
        try:
            # Some Ollama builds return JSONL (streamed) even without stream=True
            for line in resp.text.splitlines():
                try:
                    j = json.loads(line)
                    if "response" in j:
                        text_out += j["response"]
                except Exception:
                    pass
            if text_out:
                return text_out.strip()

            # Otherwise, parse normal JSON
            j = resp.json()
            if "response" in j:
                return j["response"].strip()
            return resp.text.strip()
        except Exception:
            return resp.text.strip()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
