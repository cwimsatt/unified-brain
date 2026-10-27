#!/usr/bin/env python3

# unified-brain — cheaper, faster more accurate AI coding agents, by DevLoops
# Copyright (C) 2026 DevLoops SINGLE MEMBER P.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
unified-brain — MCP server for Cursor, Claude Code and any MCP client
---------------------------------------------------------------------
One tool, `retrieve`: the agent hands over the user's question, gets back
the evidence — the code bodies that answer it, each with file, symbol and
line — and writes the answer from them. It is a thin client of the running
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