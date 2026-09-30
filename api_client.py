"""Public AI API engine: any OpenAI-compatible chat-completions endpoint.

Default provider is Pollinations AI (https://text.pollinations.ai/openai), a
public AI API that works WITHOUT an API key, so the prototype runs out of the
box. Swap providers by pointing AI_API_BASE_URL / AI_API_MODEL / AI_API_KEY at
any OpenAI-compatible gateway (Vercel AI Gateway, OpenRouter, OpenAI, ...).

Credentials are read from the environment only and are never written to disk,
never committed, and never echoed back in output or logs.
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .labels import CATEGORIES, ROUTING, SLA_HOURS, URGENCIES

DEFAULT_BASE_URL = "https://text.pollinations.ai"
DEFAULT_MODEL = "openai-fast"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "api"
JSON_SCHEMA = (
    '{"category": "billing|technical|shipping|account|other", '
    '"urgency": "low|medium|high|critical", '
    '"summary": "max 12 words", '
    '"confidence": 0.0}'
)
SYSTEM_PROMPT = (
    "You triage customer support tickets. Reply with a single JSON object that "
    "matches this schema and contains no prose, no markdown and no code fences: "
    + JSON_SCHEMA
    + ". Rules: category is the single best fit; urgency reflects business "
    "impact and time pressure (critical = outage, data loss, security or money "
    "actively bleeding; high = blocked user or deadline; low = question or "
    "feedback)."
)


class APIError(RuntimeError):
    """Raised when the remote engine cannot produce a usable triage object."""


def extract_json(raw):
    """Pull the first JSON object out of a model reply, tolerating fences.

    Brace counting skips string literals, so a stray '}' inside a summary
    string does not truncate the object.
    """
    if not raw:
        raise ValueError("empty response")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    in_string = escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced braces in response")


def coerce(payload):
    """Validate/normalise the model payload against our label schema."""
    def pick(value, allowed, default):
        token = str(value or "").strip().lower()
        if token in allowed:
            return token
        for item in allowed:  # 'Billing issues' -> billing
            if item in token:
                return item
        return default

    category_raw = str(payload.get("category") or "").strip().lower()
    urgency_raw = str(payload.get("urgency") or "").strip().lower()
    category = pick(category_raw, CATEGORIES, "other")
    urgency = pick(urgency_raw, URGENCIES, "medium")
    # 'valid' means the reply matched the schema exactly; a fuzzy label is
    # recovered into a real one but still counted as off-schema so the caller
    # can retry once and flag the ticket for a human.
    known = category_raw in CATEGORIES and urgency_raw in URGENCIES
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "category": category,
        "urgency": urgency,
        "summary": str(payload.get("summary", ""))[:160],
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "valid": known,
    }


class TriageAPIClient:
    """Thin stdlib HTTP client for an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url=None, model=None, api_key=None, timeout=45,
                 retries=3, seed=11, temperature=0.0, min_interval=None):
        self.base_url = (base_url or os.environ.get("AI_API_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("AI_API_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("AI_API_KEY", "")
        self.timeout = timeout
        self.retries = retries
        self.seed = seed
        self.temperature = temperature
        # The keyless tier is rate limited, so pace ourselves instead of
        # burning the budget on retries.
        self.min_interval = (float(os.environ.get("AI_API_MIN_INTERVAL", "3"))
                             if min_interval is None else min_interval)
        self._last_call = 0.0

    def _throttle(self):
        wait = self._last_call + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    # -- plumbing ---------------------------------------------------------
    @property
    def uses_key(self):
        return bool(self.api_key)

    @property
    def cache_enabled(self):
        return os.environ.get("AI_API_CACHE", "1") != "0"

    def _cache_path(self, body):
        """Same prompt + model + endpoint -> same file, so re-runs are stable."""
        digest = hashlib.sha1(
            json.dumps({"endpoint": self.endpoint(), "body": body},
                       sort_keys=True).encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def _cache_read(self, body):
        if not self.cache_enabled:
            return None
        path = self._cache_path(body)
        try:
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            return (cached["content"], cached.get("usage") or {})
        except (OSError, KeyError, ValueError):
            return None

    def _cache_write(self, body, content, usage):
        if not self.cache_enabled:
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path(body), "w", encoding="utf-8") as fh:
                json.dump({"content": content, "usage": usage, "at": time.time()}, fh)
        except OSError:  # a read-only filesystem must not break triage
            pass

    def endpoint(self):
        override = os.environ.get("AI_API_ENDPOINT")
        if override:
            return override
        if self.base_url == DEFAULT_BASE_URL:
            return self.base_url + "/openai"  # Pollinations' OpenAI-compatible path
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def describe(self):
        """Safe-to-print description of the remote engine (never the key)."""
        return {"engine": "public-llm-api", "endpoint": self.endpoint(),
                "model": self.model, "api_key_configured": self.uses_key}

    def _post(self, messages):
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json",
                   "User-Agent": "ticket-triage-prototype/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = json.dumps(body).encode("utf-8")
        cached = self._cache_read(body)
        if cached is not None:
            return cached[0], dict(cached[1], cached=True)
        error = APIError("request failed")
        for attempt in range(self.retries):
            self._throttle()
            try:
                request = urllib.request.Request(
                    self.endpoint(), data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):  # some gateways return content parts
                    content = "".join(part.get("text", "") for part in content)
                usage = payload.get("usage") or {}
                self._cache_write(body, content, usage)
                return content, usage
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:180]
                except Exception:  # noqa: BLE001
                    detail = ""
                error = APIError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404):
                    break  # bad request or bad credentials: retrying cannot help
                if exc.code in (402, 429) and not self.uses_key:
                    error = APIError(
                        f"HTTP {exc.code}: the anonymous tier of {self.endpoint()} is "
                        "rate limited or needs payment. Wait a minute, or point "
                        "AI_API_BASE_URL/AI_API_KEY at your own OpenAI-compatible "
                        "endpoint (see .env.example).")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = APIError(f"network error: {exc}")
            except (KeyError, ValueError) as exc:
                error = APIError(f"unexpected response shape: {exc}")
            if attempt < self.retries - 1:
                time.sleep(min(20.0, 2.0 * (2 ** attempt)))
        raise error

    # -- public API -------------------------------------------------------
    def triage(self, text, context=""):
        """Classify one ticket remotely and return the triage object."""
        started = time.time()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user = text if not context else f"{text}\n\nContext: {context}"
        messages.append({"role": "user", "content": user})

        raw, usage = self._post(messages)
        attempts = 1
        try:
            parsed = coerce(extract_json(raw))
        except ValueError:
            parsed = {"valid": False}
        if not parsed.get("valid"):
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That reply did not match the schema. "
                                            "Reply with only this JSON object: " + JSON_SCHEMA},
            ]
            raw, usage = self._post(messages)
            attempts = 2
            parsed = coerce(extract_json(raw))

        return {
            "text": text,
            "engine": "public-llm-api",
            "model": self.model,
            "category": parsed["category"],
            "category_confidence": parsed["confidence"],
            "urgency": parsed["urgency"],
            "urgency_model_guess": parsed["urgency"],
            "urgency_source": "api",
            "urgency_confidence": parsed["confidence"],
            "summary": parsed["summary"],
            "team": ROUTING[parsed["category"]],
            "sla_hours": SLA_HOURS[parsed["urgency"]],
            "requires_review": (not parsed["valid"]) or parsed["confidence"] < 0.6,
            "attempts": attempts,
            "tokens": usage,
            "latency_ms": int((time.time() - started) * 1000),
            "raw": raw[:400],
        }

    def health(self):
        """Cheap reachability probe used by the CLI before a batch run."""
        try:
            raw, _ = self._post([{"role": "user", "content": "Reply with the word PONG only."}])
            return {"reachable": True, "reply": raw.strip()[:40], **self.describe()}
        except APIError as exc:
            return {"reachable": False, "error": str(exc)[:200], **self.describe()}

