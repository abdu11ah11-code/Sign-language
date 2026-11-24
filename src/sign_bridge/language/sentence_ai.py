"""Turn a sequence of recognized signs into natural English with an AI model.

Sign languages have their own grammar: an ASL signer signs ``MOM DAD NO
DRINK`` for "Mom and Dad don't drink." The recognizer outputs the signs; this
module asks a large language model (Gemini, Claude or an OpenAI model) to
write the English sentence — and nothing more:

* the model may add only grammar (articles, "and", verb forms, tense,
  punctuation) and must not add people, objects, facts or feelings that were
  not signed;
* when the signs can be read in more than one way it returns up to
  ``max_options`` readings, best first;
* the raw signs are always kept and shown next to the AI sentence.

Without an API key, without internet, or on any error, :meth:`SentenceAI.compose`
falls back to the plain word sequence ("Mom dad no drink.") — it never raises,
so the live app keeps running.

Only the sign words are sent to the provider (no video, no landmarks).
API keys come from environment variables or ``.streamlit/secrets.toml``
(git-ignored) — see :func:`find_api_key`.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sign_bridge.logging_conf import get_logger

log = get_logger(__name__)

PROVIDERS = ("gemini", "anthropic", "openai")

#: Model used when ``ai.model`` is empty. Providers rename models often — set
#: ``ai.model`` in config.yaml if a default stops working.
#: The Gemini default is a *Flash-Lite* model: writing one short sentence is an
#: easy task, and Flash-Lite has a far bigger free daily quota than Flash.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
}

#: HTTP codes worth retrying: the provider is busy or rate-limiting us, not
#: a problem with the request itself.
RETRY_CODES = (429, 500, 502, 503, 529)
#: Extra attempts after the first, with a short pause that doubles each time.
RETRIES = 2
RETRY_PAUSE = 1.0

#: Environment variable / secrets.toml key holding each provider's API key.
KEY_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

SYSTEM_PROMPT = """\
You turn the output of a sign-language recognizer into natural English.
You receive the signs in the order they were signed (one gloss per item,
already written as English words). Sign-language grammar differs from
English: topics come first, little words are left out, negation and
questions are shown by signs like NO, NOT, WHAT, WHERE.

Rules:
1. Write what the signer most likely meant, as natural English.
2. Use ONLY the meaning of the given signs. You may add grammar words
   (a, the, and, is, do, don't, to, ...), choose tense and person, and add
   punctuation. Never add new people, objects, places, numbers, facts or
   feelings that were not signed.
3. If the signs can reasonably mean different things, give up to {n}
   different readings, best first. Otherwise give one.
4. Keep each reading short — about as long as the signs.
5. Answer ONLY with JSON: {{"sentences": ["...", "..."]}}
"""


@dataclass(frozen=True)
class AiResult:
    """Outcome of :meth:`SentenceAI.compose`.

    Attributes:
        signs: The signs that were sent (display words).
        sentences: English readings, best first (never empty).
        source: Provider that wrote them, or ``"plain"`` for the fallback.
        error: Why the fallback was used, if it was.
    """

    signs: tuple[str, ...]
    sentences: tuple[str, ...]
    source: str
    error: str | None = None

    @property
    def best(self) -> str:
        """The preferred reading."""
        return self.sentences[0]


def plain_sentence(signs: Sequence[str]) -> str:
    """The signs joined as a sentence: ``["Mom", "Dad"]`` → ``"Mom dad."``."""
    words = " ".join(s.strip() for s in signs if s.strip()).strip()
    if not words:
        return ""
    words = words[0].upper() + words[1:].lower()
    return words if words[-1] in ".!?" else words + "."


def parse_reply(text: str, max_options: int = 3) -> list[str]:
    """Pull the sentences out of a model reply (JSON, fenced JSON or plain lines)."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            items = data.get("sentences") if isinstance(data, dict) else None
            if isinstance(items, list):
                out = [str(s).strip() for s in items if str(s).strip()]
                if out:
                    return out[:max_options]
        except ValueError:
            pass
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"')
             for ln in text.splitlines()]
    return [ln for ln in lines if ln][:max_options]


# ── API keys ─────────────────────────────────────────────────────────────
def _read_secrets(path: Path) -> dict[str, Any]:
    try:
        import tomllib

        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        return {}


def find_api_key(provider: str, secrets_path: Path | None = None) -> str | None:
    """API key for ``provider`` from the environment or ``secrets.toml``."""
    name = KEY_NAMES.get(provider)
    if name is None:
        return None
    value = os.environ.get(name) or ""
    if not value and secrets_path is not None:
        value = str(_read_secrets(secrets_path).get(name) or "")
    return value.strip() or None


def save_api_key(provider: str, key: str, secrets_path: Path) -> None:
    """Store a key in ``secrets.toml`` (keeps any other entries in the file)."""
    name = KEY_NAMES[provider]
    data = {k: v for k, v in _read_secrets(secrets_path).items() if isinstance(v, str)}
    data[name] = key.strip()
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Private API keys for SignBridge — never commit this file.\n"]
    lines += [f"{k} = {json.dumps(v)}\n" for k, v in sorted(data.items())]
    secrets_path.write_text("".join(lines), encoding="utf-8")


# ── providers ────────────────────────────────────────────────────────────
def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
        return json.loads(resp.read().decode("utf-8"))


def _call_gemini(model: str, key: str, system: str, user: str, timeout: float) -> str:
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key},
        {"systemInstruction": {"parts": [{"text": system}]},
         "contents": [{"role": "user", "parts": [{"text": user}]}],
         "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}},
        timeout)
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _call_anthropic(model: str, key: str, system: str, user: str, timeout: float) -> str:
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        {"model": model, "max_tokens": 300, "temperature": 0.2, "system": system,
         "messages": [{"role": "user", "content": user}]},
        timeout)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _call_openai(model: str, key: str, system: str, user: str, timeout: float) -> str:
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {"model": model,
         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
         "response_format": {"type": "json_object"}},
        timeout)
    return str(data["choices"][0]["message"]["content"])


_CALLS = {"gemini": _call_gemini, "anthropic": _call_anthropic, "openai": _call_openai}


@dataclass
class SentenceAI:
    """Signs → natural English via an AI provider, with a safe fallback.

    Args:
        provider: ``"gemini"``, ``"anthropic"``, ``"openai"`` or ``"none"``.
        api_key: The provider's key (None → always the plain fallback).
        model: Model id; empty uses :data:`DEFAULT_MODELS`.
        max_options: Most readings to return for ambiguous signs.
        timeout: Seconds to wait for the provider.
        language: Sign language name used in the prompt.
    """

    provider: str
    api_key: str | None
    model: str = ""
    max_options: int = 3
    timeout: float = 10.0
    language: str = "ASL"
    retries: int = RETRIES
    _cache: dict[tuple[str, ...], AiResult] = field(default_factory=dict, repr=False)

    @property
    def available(self) -> bool:
        """True when a provider and key are configured."""
        return self.provider in _CALLS and bool(self.api_key)

    def compose(self, signs: Sequence[str]) -> AiResult:
        """English readings for ``signs`` (never raises; falls back to plain words)."""
        signs_t = tuple(s.strip() for s in signs if s and s.strip())
        plain = plain_sentence(signs_t)
        if not signs_t:
            return AiResult(signs_t, ("",), "plain")
        if not self.available:
            return AiResult(signs_t, (plain,), "plain", error="no API key")
        if signs_t in self._cache:
            return self._cache[signs_t]
        system = SYSTEM_PROMPT.format(n=self.max_options)
        user = (f"Sign language: {self.language}\n"
                f"Signs, in order: {json.dumps([s.upper() for s in signs_t])}")
        model = self.model or DEFAULT_MODELS[self.provider]
        pause = RETRY_PAUSE
        for attempt in range(self.retries + 1):
            try:
                reply = _CALLS[self.provider](model, str(self.api_key), system, user, self.timeout)
                sentences = parse_reply(reply, self.max_options)
                if not sentences:
                    raise ValueError("empty reply")
                result = AiResult(signs_t, tuple(sentences), self.provider)
                self._cache[signs_t] = result
                log.info("ai_sentence", provider=self.provider, signs=len(signs_t),
                         attempt=attempt + 1)
                return result
            except urllib.error.HTTPError as exc:
                retry = exc.code in RETRY_CODES
                reason = f"{self.provider} error {exc.code}"
                if exc.code in (401, 403):
                    reason += " — check the API key"
                elif exc.code == 404:
                    reason += f" — model '{model}' not found; set ai.model in config.yaml"
                elif exc.code == 429:
                    reason += " — rate limit / free quota reached, try again shortly"
                elif exc.code in (500, 502, 503, 529):
                    reason += " — the AI service is busy; try again in a moment"
            except (urllib.error.URLError, TimeoutError, OSError):
                reason, retry = "no connection to the AI service", True
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                reason, retry = f"unexpected reply ({type(exc).__name__})", False
            if not retry or attempt == self.retries:
                break
            log.info("ai_sentence_retry", provider=self.provider, reason=reason,
                     attempt=attempt + 1)
            time.sleep(pause)
            pause *= 2
        log.warning("ai_sentence_fallback", provider=self.provider, reason=reason)
        return AiResult(signs_t, (plain,), "plain", error=reason)
