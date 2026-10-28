#!/usr/bin/env python3

# unified-brain — cheaper, faster more accurate AI coding agents, by DevLoops
# Copyright (C) 2026 DevLoops SINGLE MEMBER P.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
unified-brain — MCP server for Cursor, Claude Code and any MCP client
---------------------------------------------------------------------
Two tools. `retrieve`: the agent hands over the user's question, gets back
the evidence — the code bodies that answer it, each with file, symbol and
line — and writes the answer from them. `evaluate_claims`: the agent states
claims about the code and gets each back as true, false, conflicted or
unknown with the passages behind it. Both are thin clients of the running
service (python codebase_qa_service.py); the service holds the database
file and the API keys.

    UNIFIED_BRAIN_URL   where the service runs (default http://127.0.0.1:8000)

Claude Code (.mcp.json in the project, or ~/.claude.json):
    {"mcpServers": {"unified-brain": {
        "command": "/path/to/unified-brain/.venv/bin/python",
        "args": ["/path/to/unified-brain/unified_brain_mcp.py"],
        "env": {"UNIFIED_BRAIN_URL": "http://127.0.0.1:8000"}}}}

Cursor (.cursor/mcp.json): the same object.
"""
import json
import os

import httpx
try:                                   # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

URL = os.environ.get("UNIFIED_BRAIN_URL", "http://127.0.0.1:8000").rstrip("/")

mcp = _Server(
    "unified-brain",
    instructions=(
        "unified-brain has already read the whole repository and its git "
        "history. For any question about how the code works, call `retrieve` "
        "ONCE with the user's question as written, then answer from the "
        "evidence it returns. Do not split the question, do not call it "
        "again for parts of it, and do not grep or open files to verify: the "
        "bodies are the source text, and the `note` field says whether the "
        "evidence is complete."),
)


def _retrieve(question: str, from_commit: str = "", to_commit: str = "",
              post=httpx.post) -> dict:
    """POST /api/retrieve and shape the reply for the agent. `post` is the
    HTTP call (tests pass a stub). Never raises: every failure is
    {"error": ...}; the `note` says honestly what the evidence is."""
    try:
        r = post(f"{URL}/api/retrieve",
                 json={"question": question, "from_commit": from_commit,
                       "to_commit": to_commit},
                 timeout=1800)
    except httpx.HTTPError as e:
        return {"error": f"unified-brain service not reachable at {URL}: {e}"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except (ValueError, AttributeError):
            detail = r.text
        return {"error": f"HTTP {r.status_code}: {detail}"}
    try:
        d = r.json()
    except ValueError:
        d = None
    if not isinstance(d, dict):
        return {"error": f"unified-brain service answered HTTP 200 with no JSON object: "
                         f"{(r.text or '')[:300]}"}
    evidence = d.get("evidence") or []
    n = len(evidence)
    info = d.get("info") or {}
    sel = (info.get("select") or {}) if isinstance(info, dict) else {}
    scored = sel.get("scored") or 0
    failed = sel.get("failed") or 0
    candidates = sel.get("candidates") or 0
    log = d.get("log") or []
    # The name-following pass reports its failure only as a log line.
    expand_failed = any(isinstance(m, str) and m.startswith("Expansion failed")
                        for m in log)
    if n == 0 and candidates == 0:
        note = ("No evidence: no symbol was scored (0 candidates: empty index "
                "or window). Say so; do not guess.")
    elif n == 0:
        note = (f"No evidence: Jev kept no body above the score cut "
                f"({scored} scored, {failed} failed). Say so; do not guess.")
    elif failed > 0 or expand_failed:
        why = []
        if failed > 0:
            why.append(f"{failed} of {candidates} judge calls failed")
        if expand_failed:
            why.append("the name-following pass failed")
        note = (f"Evidence: {n} bodies, full code included; {' and '.join(why)}, "
                "so the set may be incomplete.")
    else:
        note = (f"Complete evidence: {n} bodies, full code included. "
                "Answer from these; nothing to fetch or verify.")
    return {
        "note": note,
        "window": d.get("window", ""),
        "evidence": evidence,
        "leads": d.get("leads") or [],
        "info": info,
        "log": log,
    }


def _parse_claims(claims) -> list:
    """["name: description", ...] (or dicts) -> [{"name", "description"}]."""
    out = []
    for c in claims or []:
        if isinstance(c, dict) and c.get("name") and c.get("description"):
            out.append({k: c[k] for k in ("name", "description", "params", "world", "scope") if k in c})
            continue
        if not isinstance(c, str) or ":" not in c:
            raise ValueError(f"claim {c!r}: write it as \"name: description\"")
        name, desc = c.split(":", 1)
        if not name.strip() or not desc.strip():
            raise ValueError(f"claim {c!r}: write it as \"name: description\"")
        out.append({"name": name.strip(), "description": desc.strip()})
    if not out:
        raise ValueError("give at least one claim")
    return out


def _evaluate_claims(question: str, claims, rules: str = "", from_commit: str = "",
                     to_commit: str = "", per_symbol: bool = False, post=httpx.post) -> dict:
    """POST /api/claims/evaluate and shape the reply for the agent: one line
    per conclusion with its value and the passages behind it (per symbol when
    `per_symbol`). Never raises."""
    try:
        parsed = _parse_claims(claims)
        if per_symbol:
            for c in parsed:
                c.setdefault("scope", "symbol")
        body = {"question": question, "claims": parsed,
                "rules": json.loads(rules) if (rules or "").strip() else [],
                "from_commit": from_commit, "to_commit": to_commit}
    except ValueError as e:
        return {"error": f"bad input: {e}"}
    try:
        r = post(f"{URL}/api/claims/evaluate", json=body, timeout=1800)
    except httpx.HTTPError as e:
        return {"error": f"unified-brain service not reachable at {URL}: {e}"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except (ValueError, AttributeError):
            detail = r.text
        return {"error": f"HTTP {r.status_code}: {detail}"}
    try:
        d = r.json()
    except ValueError:
        d = None
    if not isinstance(d, dict) or not isinstance(d.get("judgments"), dict):
        return {"error": f"unified-brain service answered HTTP 200 with no evaluation: "
                         f"{(r.text or '')[:300]}"}
    cls = ((d.get("info") or {}).get("classification") or {})
    passages, failed, jobs = cls.get("passages") or 0, cls.get("failed") or 0, cls.get("jobs") or 0
    def compact(v):
        if v.get("scope") == "symbol":
            return {"scope": "symbol", "summary": v.get("summary"),
                    "by_symbol": [{kk: r.get(kk) for kk in ("file", "symbol", "line", "value")}
                                  for r in (v.get("by_symbol") or [])]}
        return {"value": v.get("value"), "because": v.get("because"),
                "sources": [{kk: s.get(kk) for kk in ("file", "symbol", "line", "id")}
                            for s in (v.get("sources") or [])]}
    judgments = {k: compact(v) for k, v in d["judgments"].items()}
    if passages == 0:
        note = ("No evidence: nothing was retrieved for the question, so every claim "
                "is unknown. Say so; do not guess.")
    elif failed:
        note = (f"{failed} of {jobs} judge calls failed, so conclusions may rest on "
                f"partial evidence ({passages} passages read).")
    else:
        note = (f"Evaluated {len(judgments)} conclusions over {passages} passages. "
                "true/false: the passages support/contradict it; unknown: no passage "
                "spoke to it; conflicted: passages disagree (see sources).")
    return {"note": note, "id": d.get("id"), "window": d.get("window", ""),
            "judgments": judgments, "cost_usd": d.get("cost_usd")}


@mcp.tool()
def evaluate_claims(question: str, claims: list[str], rules: str = "",
                    from_commit: str = "", to_commit: str = "", per_symbol: bool = False) -> str:
    """Check specific claims about the code against the evidence.

    Give the question that finds the relevant code and the claims to assess,
    each as "name: description" (e.g. "validates_input: The function rejects
    an empty name before using it"). Every retrieved body is read by the
    judge model for each claim; each conclusion comes back as true, false,
    conflicted or unknown with the file, symbol and line of the passages it
    rests on. Optional `rules` (a JSON list) combine claims:
    [{"head": "safe", "iff": {"and": ["validates_input", {"not": "logs_secret"}]}}].
    With per_symbol=true each claim is judged for every retrieved symbol
    from its own body, and the reply lists file:symbol:line = value rows
    ("which of these functions validate input"). Report the values as given;
    unknown means the evidence was silent, not that the claim is false. Takes
    20–90 seconds.
    """
    return json.dumps(_evaluate_claims(question, claims, rules, from_commit, to_commit, per_symbol),
                      ensure_ascii=False, indent=1)


@mcp.tool()
def retrieve(question: str, from_commit: str = "", to_commit: str = "") -> str:
    """Get the evidence that answers a question about the repository.

    Call this ONCE, with the user's whole question as they wrote it. It runs
    the full search (embeddings → a judge model scores every candidate →
    the judge follows names through the code) and returns every body that
    answers the question, with file, symbol, line and the code in full.
    Answer from that evidence; do not call again for sub-questions and do
    not re-read the files — the bodies are the code. Takes 20–60 seconds.
    Give from_commit and to_commit to see the code as it was at to_commit.
    The `note` field says whether the evidence is complete; repeat what it
    says when it is not.
    """
    return json.dumps(_retrieve(question, from_commit, to_commit),
                      ensure_ascii=False, indent=1)


if __name__ == "__main__":
    mcp.run()