"""Thin provider-agnostic LLM client (Anthropic / Google Gemini / OpenAI) using plain HTTP."""
import json
import re
import time

import requests

from config import Config


class LLMError(Exception):
    pass


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_call = 0.0

    # ---------- public ----------
    def complete(self, system: str, user_text: str, images: list[tuple[str, str]] | None = None,
                 max_tokens: int = 8192) -> str:
        images = images or []
        provider = self.cfg.provider
        if provider == "anthropic":
            url, headers, body = self._anthropic(system, user_text, images, max_tokens)
        elif provider == "openai":
            url, headers, body = self._openai(system, user_text, images, max_tokens)
        else:
            url, headers, body = self._google(system, user_text, images, max_tokens)

        last_err = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout)
            except requests.RequestException as e:
                last_err = f"network error: {e}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code in RETRY_STATUS:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                wait = resp.headers.get("retry-after")
                try:
                    delay = float(wait) if wait else min(2 ** attempt * 2, 60)
                except ValueError:
                    delay = 10
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                text = self._extract_text(resp.json())
            except (ValueError, KeyError, IndexError, TypeError) as e:
                last_err = f"bad response shape: {e}"
                time.sleep(2)
                continue
            if text.strip():
                return text
            last_err = "empty response"
        raise LLMError(f"LLM call failed after retries: {last_err}")

    # ---------- helpers ----------
    def _throttle(self):
        gap = self.cfg.min_call_interval - (time.time() - self._last_call)
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.time()

    def _anthropic(self, system, user_text, images, max_tokens):
        content = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": d}} for m, d in images]
        content.append({"type": "text", "text": user_text})
        body = {"model": self.cfg.model, "max_tokens": min(max_tokens, 8192), "temperature": 0,
                "system": system, "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": self.cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return "https://api.anthropic.com/v1/messages", headers, body

    def _openai(self, system, user_text, images, max_tokens):
        content = [{"type": "text", "text": user_text}]
        for m, d in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:{m};base64,{d}", "detail": "high"}})
        body = {"model": self.cfg.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        if self.cfg.model.startswith("gpt-5"):
            body["max_completion_tokens"] = max(max_tokens, 12000)  # reasoning tokens count too
            body["reasoning_effort"] = "low"
        else:
            body["max_tokens"] = min(max_tokens, 16000)
            body["temperature"] = 0
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "content-type": "application/json"}
        return "https://api.openai.com/v1/chat/completions", headers, body

    def _google(self, system, user_text, images, max_tokens):
        parts = [{"inlineData": {"mimeType": m, "data": d}} for m, d in images]
        parts.append({"text": user_text})
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max(max_tokens, 12000),
                                     "responseMimeType": "application/json"}}
        headers = {"x-goog-api-key": self.cfg.api_key, "content-type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.cfg.model}:generateContent"
        return url, headers, body

    def _extract_text(self, data: dict) -> str:
        p = self.cfg.provider
        if p == "anthropic":
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        if p == "openai":
            return data["choices"][0]["message"]["content"] or ""
        cands = data.get("candidates") or []
        if not cands:
            raise KeyError(f"no candidates: {str(data)[:200]}")
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(x.get("text", "") for x in parts if not x.get("thought"))

    # ---------- JSON helper ----------
    def complete_json(self, system, user_text, images=None, max_tokens=8192) -> dict:
        text = self.complete(system, user_text, images, max_tokens)
        return parse_json(text)


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s == -1 or e <= s:
            raise LLMError(f"No JSON object in model output: {text[:200]}")
        try:
            obj = json.loads(t[s:e + 1])
        except json.JSONDecodeError as err:
            raise LLMError(f"Invalid JSON from model: {err}") from err
    if not isinstance(obj, dict):
        raise LLMError("Model output JSON is not an object")
    return obj
