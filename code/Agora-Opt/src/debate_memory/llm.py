"""
Lightweight HTTP client for OpenAI-compatible chat completions.

- Credentials are read from environment variables only.
- Supported environment variables:
    * `LLM_API_BASE_URL`
    * `LLM_API_KEY`
    * `OPENAI_BASE_URL`
    * `OPENAI_API_KEY`
    * `API_URL`
    * `API_KEY`
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import requests

def _get_credentials() -> Dict[str, str]:
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )
    base_url = (
        os.getenv("LLM_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("API_URL")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set one of: LLM_API_KEY, OPENAI_API_KEY, API_KEY."
        )
    if not base_url:
        raise RuntimeError(
            "Missing API base URL. Set one of: "
            "LLM_API_BASE_URL, OPENAI_BASE_URL, API_URL."
        )
    return {"api_key": api_key, "base_url": base_url.rstrip("/")}


def _post_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict:
    creds = _get_credentials()
    url = f"{creds['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Non-JSON response from LLM API: {response.text[:200]}") from exc


def _extract_content(result: Dict) -> str:
    choices = result.get("choices")
    if not choices:
        raise RuntimeError(f"LLM API response missing 'choices': {result}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"LLM API response missing message content: {result}")
    return content


def get_response(prompt: str, model: str, temperature: float = 0.01, maximum_retries: int = 10) -> str:
    """
    Send a chat completion request using OpenAI-compatible REST calls.
    """
    if model.startswith("deepseek"):
        real_model = model.replace("-chat", "-v3").replace("-reasoner", "-r1")
    else:
        real_model = model

    attempts = max(1, maximum_retries)
    last_error: Exception | None = None
    while attempts > 0:
        try:
            result = _post_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=real_model,
                temperature=temperature,
                max_tokens=16384,
            )
            return _extract_content(result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts -= 1
            if attempts == 0:
                break
            print(f"Error using API: {exc}. Retrying...")
            time.sleep(2)

    raise RuntimeError(f"Failed to get response from API after retries: {last_error}")
