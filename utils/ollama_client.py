

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
    Call Ollama's /api/generate endpoint with improved error handling.
    Returns the generated text as a string.
    """
    base = base_url or DEFAULT_OLLAMA_BASE
    key = api_key if api_key is not None else DEFAULT_OLLAMA_API_KEY
    url = f"{base.rstrip('/')}/api/generate"

    # Use Ollama's native parameters
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # IMPORTANT: Disable streaming for simpler handling
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
        }
    }

    headers = _build_headers(key)

    try:
        print(f"[Ollama] Calling {model} with prompt length: {len(prompt)} chars...")
        
        resp = requests.post(
            url, 
            json=payload, 
            headers=headers, 
            timeout=timeout
        )
        resp.raise_for_status()

        # Parse response
        try:
            result = resp.json()
            
            # Check if response key exists
            if "response" in result:
                generated_text = result["response"].strip()
                print(f"[Ollama] Generated {len(generated_text)} chars")
                return generated_text
            
            # Check for error
            if "error" in result:
                raise RuntimeError(f"Ollama error: {result['error']}")
            
            # Fallback: return entire response as string
            print("[Ollama] Warning: Unexpected response format")
            return str(result)
            
        except json.JSONDecodeError as e:
            print(f"[Ollama] JSON decode error: {e}")
            # Try to extract text from raw response
            text = resp.text.strip()
            if text:
                return text
            raise RuntimeError(f"Invalid JSON response from Ollama: {resp.text[:200]}")
            
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s. "
            f"Try: 1) Increase timeout, 2) Use smaller model, 3) Reduce prompt length"
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base}. "
            f"Make sure Ollama is running. Error: {e}"
        )
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_detail = resp.json().get("error", str(e))
        except:
            error_detail = resp.text[:200]
        raise RuntimeError(f"Ollama HTTP error: {error_detail}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def call_ollama_chat(
    messages: list,
    model: str = "llama3",
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 60,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Call Ollama's /api/chat endpoint (alternative for conversational format).
    """
    base = base_url or DEFAULT_OLLAMA_BASE
    key = api_key if api_key is not None else DEFAULT_OLLAMA_API_KEY
    url = f"{base.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    headers = _build_headers(key)

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        
        result = resp.json()
        if "message" in result and "content" in result["message"]:
            return result["message"]["content"].strip()
        
        return str(result)
        
    except Exception as e:
        raise RuntimeError(f"Ollama chat request failed: {e}")