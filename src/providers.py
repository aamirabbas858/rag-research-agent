"""
providers.py — where the model calls actually go.

This project ran entirely on Gemini until its prepaid balance emptied, at
which point every request returned RESOURCE_EXHAUSTED and the app stopped
working. Nothing about the agent was wrong; it just had exactly one provider.

So there are two now, tried in order, each skipped when its key is absent.
Adding a third is an environment variable rather than a code change.

--- chat and embeddings are not the same problem ---

Swapping the chat model changes the wording of an answer. Swapping the
EMBEDDING model invalidates every vector already stored: the numbers mean
different things and are usually a different length. Querying a store built
with one model using vectors from another does not raise an error — it
quietly returns the wrong chunks, and the answer on top of them looks fine.

Which is why the collection name carries the embedding model. Change the
model and you get a new, empty collection instead of a corrupted old one.
"""

from __future__ import annotations

import os

import requests

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Sent to the outside world as part of the collection name, so it must stay
# short and filesystem-safe.
EMBED_MODEL = "mistral-embed" if MISTRAL_API_KEY else "gemini-embedding-2"


class NoProviderConfigured(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "No model provider configured. Set MISTRAL_API_KEY (free tier, no card) "
            "or GEMINI_API_KEY in your .env"
        )


# ─────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────

def _mistral_chat(prompt: str) -> str:
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _gemini_chat(prompt: str) -> str:
    # Several model names because Google retires them without much notice, and
    # a 404 on one should not take the agent down.
    for model in ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"):
        r = requests.post(
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
            timeout=90,
        )
        # 404 is a retired model, 503 is a busy one. Both are worth trying the
        # next name for. Anything else is about this request, not this model.
        if r.status_code in (404, 503):
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("no Gemini model answered")


def call_llm(prompt: str) -> str:
    """First provider with a key that answers. Errors only when none do."""
    errors: list[str] = []

    for name, key, fn in (
        ("mistral", MISTRAL_API_KEY, _mistral_chat),
        ("gemini", GEMINI_API_KEY, _gemini_chat),
    ):
        if not key:
            continue
        try:
            return fn(prompt)
        except Exception as e:  # noqa: BLE001 — the next provider is the handler
            errors.append(f"{name}: {e}")

    if not errors:
        raise NoProviderConfigured()
    raise RuntimeError("Every provider failed — " + "; ".join(errors))


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    One vector for one piece of text.

    No failover here, deliberately. Falling back to a different embedding
    model mid-collection would put vectors of two different shapes and
    meanings into the same store, and the resulting nonsense would be silent.
    Better to fail loudly than to retrieve confidently wrong chunks.
    """
    if MISTRAL_API_KEY:
        r = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-embed", "input": [text]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    if GEMINI_API_KEY:
        r = requests.post(
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}",
            json={"content": {"parts": [{"text": text}]}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]["values"]

    raise NoProviderConfigured()
