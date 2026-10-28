#!/usr/bin/env python3

# unified-brain — cheaper, faster more accurate AI coding agents, by DevLoops
# Copyright (C) 2026 DevLoops SINGLE MEMBER P.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
jev_client — the one Jev call helper
------------------------------------
Jev (Typesafe's System One decision model) is reached directly at Typesafe or
through OpenRouter; both answer the same request shape. Everything in the
service that talks to Jev goes through `_jev_post`, and so does the claims
layer (factbag_jev.py), which is why these helpers live in their own module:
importing them must not require DB_PATH or start the service.

    _jev_provider()      {name, url, key, key_env, model} from the environment
    _jev_key_missing()   "" or the message telling the caller which key to set
    _jev_client()        an httpx.Client for a batch of calls (caller closes it)
    _jev_cost(data)      USD of one answered call
    _jev_post(...)       one call with retries -> (answers, cost, err)
    _scrub(text)         redacts credential shapes from log/error strings

The service imports these names, so `codebase_qa_service._jev_post` is still
the seam the tests stub.
"""
import os
import re
import time

import httpx


# Applied in order. Add a (pattern, replacement) line to extend coverage.
_SCRUB_PATTERNS = [
    # Passwords embedded in connection URIs: redis(s), bolt, neo4j(+s)
    (re.compile(r'((?:redis|rediss|bolt|neo4j)(?:\+s|\+ssc)?)://[^:/@\s]+:[^@\s]+@'),
     r'\1://***:***@'),
    # Anthropic / DeepSeek / OpenAI secret keys
    (re.compile(r'sk-(?:ant-)?[A-Za-z0-9_\-]{20,}'), 'sk-***'),
    # DevLoops API keys
    (re.compile(r'dl_(live|test)_[A-Za-z0-9_\-]+'), r'dl_\1_***'),
    # JWTs (header.payload.signature)
    (re.compile(r'eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+'), '***JWT***'),
    # Bearer tokens
    (re.compile(r'(?i)(bearer)\s+[A-Za-z0-9._\-]+'), r'\1 ***'),
    # Bare credential fields: password=…, "secret": "…", p_value: '…', etc.
    (re.compile(
        r'(?i)\b(redis_password|neo4j_password|password|passwd|secret|token|api[_-]?key|p_value)\b'
        r'(\s*["\']?\s*[:=]\s*["\']?)([^"\'\s,;}]+)'),
     r'\1\2***'),
]

def _scrub(text: str) -> str:
    """Redact known credential shapes from log/error strings.

    Covers URI passwords (redis/rediss/bolt/neo4j[+s]), sk-/dl_ keys, JWTs,
    Bearer tokens, and bare password/secret/token/p_value fields. Over-redaction
    is deliberate: a leaked secret costs more than a redacted log line.
    """
    for pat, repl in _SCRUB_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ── Jev (the judge) ──────────────────────────────────────────────────────────
# One decisions API, reached directly at Typesafe or through OpenRouter.
# JEV_PROVIDER picks the provider; unset, it is typesafe when only
# TYPESAFE_API_KEY is set and openrouter otherwise (upstream behaviour).
_JEV_PROVIDERS = {
    "typesafe":   {"url": "https://api.typesafe.ai/v1/systemone",
                   "key_env": "TYPESAFE_API_KEY",   "model": "jev-1.13.0"},
    "openrouter": {"url": "https://openrouter.ai/api/alpha/decisions",
                   "key_env": "OPENROUTER_API_KEY", "model": "typesafe/jev-1.13"},
}
# The same model under the other provider's name, so a JEV_MODEL from an old
# .env keeps working after a switch. jev-latest (Typesafe) has no alias.
_JEV_MODEL_ALIASES = {
    "typesafe":   {"typesafe/jev-1.13": "jev-1.13.0"},
    "openrouter": {"jev-1.13.0": "typesafe/jev-1.13"},
}


def _jev_provider() -> dict:
    """{name, url, key, key_env, model}, read from the environment on every
    call. JEV_API_KEY overrides the provider's key, JEV_API_BASE its url,
    JEV_MODEL its model (alias-mapped between the providers)."""
    name = (os.environ.get("JEV_PROVIDER") or "").strip().lower()
    if not name:
        name = ("typesafe" if os.environ.get("TYPESAFE_API_KEY")
                and not os.environ.get("OPENROUTER_API_KEY") else "openrouter")
    if name not in _JEV_PROVIDERS:
        raise ValueError(f"JEV_PROVIDER={name!r}: expected one of "
                         f"{', '.join(sorted(_JEV_PROVIDERS))}")
    spec = _JEV_PROVIDERS[name]
    model = os.environ.get("JEV_MODEL") or spec["model"]
    return {"name":    name,
            "url":     os.environ.get("JEV_API_BASE") or spec["url"],
            "key":     os.environ.get("JEV_API_KEY") or os.environ.get(spec["key_env"], ""),
            "key_env": spec["key_env"],
            "model":   _JEV_MODEL_ALIASES[name].get(model, model)}


def _jev_key_missing() -> str:
    """"" when Jev has an API key, else the message for the caller."""
    prov = _jev_provider()
    if prov["key"]:
        return ""
    return ("Jev needs an API key on the server (Jev gathers the evidence): "
            f"set {prov['key_env']} for provider {prov['name']}, or JEV_API_KEY. "
            "TYPESAFE_API_KEY alone selects typesafe, OPENROUTER_API_KEY alone "
            "selects openrouter; JEV_PROVIDER picks explicitly.")


def _jev_client(max_connections: int = 64, timeout: float = 60) -> httpx.Client:
    """One httpx client for a batch of Jev calls. The caller closes it."""
    return httpx.Client(
        timeout=timeout,
        limits=httpx.Limits(max_connections=max_connections,
                            max_keepalive_connections=max_connections),
        headers={"Authorization": f"Bearer {_jev_provider()['key']}"})


def _jev_cost(data: dict) -> float:
    """USD of one call. OpenRouter reports usage.cost; Typesafe reports the
    tokens only, priced at JEV_PRICE_PER_MTOK_IN per million input tokens."""
    usage = data.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage.get("cost") or 0)
    price = float(os.environ.get("JEV_PRICE_PER_MTOK_IN", "0.042"))
    return float(usage.get("input_tokens") or 0) * price / 1e6


def _jev_post(client: httpx.Client, state: dict, questions: dict, *,
              what: str, attempts: int = 4):
    """One Jev call with retries. Returns (answers, cost, err): the
    response's `answers` dict and the cost in USD on success (err None);
    (None, 0.0, the last error as text) on failure. Retries connection
    errors, 429, 529 and every 5xx: Retry-After seconds when the response
    names them (capped at 60), else min(2**attempt, 30); no sleep after the
    last try. A Typesafe HTTP 400 for max_tokens_exceeded (32k tokens per
    question) is retried once with the long-context header. `what` names
    the call in the log lines."""
    prov = _jev_provider()
    payload = {"model": prov["model"], "state": state, "questions": questions}
    headers = None
    long_ctx = False
    err = None
    attempt = 0
    while attempt < attempts:
        last = attempt >= attempts - 1
        try:
            res = client.post(prov["url"], json=payload, headers=headers)
        except httpx.HTTPError as e:
            err = f"{type(e).__name__}: {_scrub(str(e))[:200]}"
            if not last:
                time.sleep(min(2 ** attempt, 30))
            attempt += 1
            continue
        if res.status_code == 429 or res.status_code >= 500:
            err = f"HTTP {res.status_code} {_scrub(res.text[:200])}"
            if attempt == 0:
                print(f"⏳ Jev {what} call got HTTP {res.status_code}, retrying")
            if not last:
                ra = (res.headers.get("Retry-After") or "").strip()
                # isdecimal + a length bound: isdigit() also accepts '²' and
                # thousand-digit runs, which int() rejects with ValueError.
                time.sleep(min(int(ra), 60) if ra.isdecimal() and len(ra) <= 6
                           else min(2 ** attempt, 30))
            attempt += 1
            continue
        if (res.status_code == 400 and prov["name"] == "typesafe" and not long_ctx
                and "max_tokens_exceeded" in res.text):
            # Typesafe caps a question at 32k tokens unless this header is sent.
            long_ctx = True
            headers = {"TypeSafe-Danger-Zone-Long-Context": "true"}
            print(f"⏳ Jev {what} call: max_tokens_exceeded, retrying with long context")
            continue                      # not counted as an attempt
        try:
            data = res.json()
        except ValueError:
            return None, 0.0, f"HTTP {res.status_code}, not JSON: {_scrub(res.text[:200])}"
        if res.status_code != 200 or not isinstance(data, dict):
            return None, 0.0, f"HTTP {res.status_code} {_scrub(res.text[:200])}"
        return (data.get("answers") or {}), _jev_cost(data), None
    return None, 0.0, err
