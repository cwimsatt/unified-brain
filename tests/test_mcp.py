"""unified_brain_mcp._retrieve: the note the agent reads, shaped from the
service's reply, with the HTTP call stubbed."""
import json

import httpx
import pytest

import unified_brain_mcp as ub


class _Resp:
    def __init__(self, status, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def _post_returning(resp):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(resp, Exception):
            raise resp
        return resp
    post.calls = calls
    return post


def _reply(evidence, select=None, **extra):
    return {"question": "q", "window": "", "evidence": evidence, "leads": [],
            "log": ["Pre-filter off"], "info": {"select": select or {}}, **extra}


ROW = {"def_key": "a.py:f:1:FunctionDef", "file": "a.py", "symbol": "f", "code": "def f(): pass"}


def test_posts_the_question_to_the_service():
    post = _post_returning(_Resp(200, _reply([ROW], {"candidates": 3, "scored": 3, "failed": 0})))
    out = ub._retrieve("why?", "aaa", "bbb", post=post)
    (c,) = post.calls
    assert c["url"] == f"{ub.URL}/api/retrieve"
    assert c["json"] == {"question": "why?", "from_commit": "aaa", "to_commit": "bbb"}
    assert c["timeout"] == 1800
    assert set(out) == {"note", "window", "evidence", "leads", "info", "log"}
    assert out["evidence"] == [ROW] and out["log"] == ["Pre-filter off"]
    assert out["info"]["select"]["scored"] == 3


def test_note_complete():
    post = _post_returning(_Resp(200, _reply([ROW, ROW], {"candidates": 9, "scored": 9, "failed": 0})))
    note = ub._retrieve("q", post=post)["note"]
    assert note.startswith("Complete evidence: 2 bodies")
    assert "nothing to fetch or verify" in note


def test_note_incomplete_when_calls_failed():
    post = _post_returning(_Resp(200, _reply([ROW], {"candidates": 9, "scored": 7, "failed": 2})))
    note = ub._retrieve("q", post=post)["note"]
    assert note.startswith("Evidence: 1 bodies")
    assert "2 of 9 judge calls failed" in note and "incomplete" in note
    assert "Complete" not in note


def test_note_no_evidence():
    post = _post_returning(_Resp(200, _reply([], {"candidates": 9, "scored": 9, "failed": 0})))
    note = ub._retrieve("q", post=post)["note"]
    assert note.startswith("No evidence")
    assert "9 scored, 0 failed" in note and "do not guess" in note
    # no info at all: still honest, and "0 scored" is not "kept no body"
    post = _post_returning(_Resp(200, {"evidence": []}))
    out = ub._retrieve("q", post=post)
    assert out["note"].startswith("No evidence: no symbol was scored")
    assert "0 candidates" in out["note"] and "do not guess" in out["note"]
    assert "score cut" not in out["note"]
    assert out["evidence"] == [] and out["leads"] == [] and out["log"] == [] and out["info"] == {}


def test_note_no_candidates_says_nothing_was_scored():
    """An empty index or window: _jev_select reports candidates=scored=0 and
    the service's 502 guard does not fire."""
    post = _post_returning(_Resp(200, _reply([], {"candidates": 0, "scored": 0, "failed": 0})))
    note = ub._retrieve("q", post=post)["note"]
    assert note.startswith("No evidence: no symbol was scored")
    assert "0 candidates" in note and "do not guess" in note


def test_note_incomplete_when_expansion_failed():
    """JEV_EXPAND=1 (the default): a failed name-following pass reaches the
    client only as a log line, and info has no `expand` entry."""
    log = ["Pre-filter off", "Expansion failed, continuing without it: HTTP 500"]
    post = _post_returning(_Resp(200, _reply([ROW], {"candidates": 9, "scored": 9, "failed": 0}, log=log)))
    out = ub._retrieve("q", post=post)
    assert out["note"].startswith("Evidence: 1 bodies")
    assert "name-following pass failed" in out["note"] and "incomplete" in out["note"]
    assert "Complete" not in out["note"] and out["log"] == log
    # both at once: both reasons named
    post = _post_returning(_Resp(200, _reply([ROW], {"candidates": 9, "scored": 7, "failed": 2}, log=log)))
    note = ub._retrieve("q", post=post)["note"]
    assert "2 of 9 judge calls failed and the name-following pass failed" in note
    # a successful expansion keeps the complete verdict
    post = _post_returning(_Resp(200, _reply([ROW], {"candidates": 9, "scored": 9, "failed": 0},
                                              log=["Expansion added 0 leads over 1 hop(s)"])))
    assert ub._retrieve("q", post=post)["note"].startswith("Complete evidence")


def test_http_error_with_detail():
    out = ub._retrieve("q", post=_post_returning(_Resp(502, {"detail": "Jev scored none of 9 candidates: HTTP 401"})))
    assert out == {"error": "HTTP 502: Jev scored none of 9 candidates: HTTP 401"}
    out = ub._retrieve("q", post=_post_returning(_Resp(500, text="Internal Server Error")))
    assert out == {"error": "HTTP 500: Internal Server Error"}
    out = ub._retrieve("q", post=_post_returning(_Resp(400, ["not", "an", "object"])))
    assert out["error"].startswith("HTTP 400:")


def test_non_json_200_is_an_error():
    out = ub._retrieve("q", post=_post_returning(_Resp(200, text="<html>proxy page</html>")))
    assert set(out) == {"error"} and "no JSON object" in out["error"] and "proxy page" in out["error"]
    out = ub._retrieve("q", post=_post_returning(_Resp(200, ["list"])))
    assert set(out) == {"error"} and "no JSON object" in out["error"]


def test_unreachable_service_is_an_error():
    out = ub._retrieve("q", post=_post_returning(httpx.ConnectError("connection refused")))
    assert set(out) == {"error"}
    assert "not reachable" in out["error"] and ub.URL in out["error"]


def test_tool_wrapper_returns_json_text(monkeypatch):
    monkeypatch.setattr(ub, "_retrieve", lambda q, f="", t="", post=None: {"note": "n", "q": q, "f": f, "t": t})
    assert callable(ub.retrieve)
    out = json.loads(ub.retrieve("q", "a", "b"))
    assert out == {"note": "n", "q": "q", "f": "a", "t": "b"}


def test_server_instructions_point_at_the_note():
    assert "`note`" in ub.mcp.instructions
    assert "`note`" in ub.retrieve.__doc__


# ── evaluate_claims ─────────────────────────────────────────────────────────

def _claims_reply(judgments, passages=2, failed=0, jobs=4, **extra):
    return {"id": "abc", "window": "", "judgments": judgments, "cost_usd": 0.0008,
            "info": {"classification": {"passages": passages, "failed": failed, "jobs": jobs}}, **extra}


J = {"returns_greeting": {"value": "true", "because": "evidence",
                          "sources": [{"id": "pkg%2Fapp.py%3Ahello%3A4%3AFunctionDef#0-40", "file": "pkg/app.py",
                                       "symbol": "hello", "line": 4, "struct": "FunctionDef"}],
                          "explanation": {"claim": "returns_greeting() = true"}},
     "ready": {"value": "unknown", "because": "inconclusive rules", "sources": [], "explanation": {}}}


def test_evaluate_claims_posts_parsed_claims_and_rules():
    post = _post_returning(_Resp(200, _claims_reply(J)))
    out = ub._evaluate_claims("how does hello greet?",
                              ["returns_greeting: The function returns a greeting string.",
                               {"name": "reviewed", "description": "Signed off.", "world": "closed"}],
                              rules='[{"head": "ready", "iff": {"and": ["returns_greeting", "reviewed"]}}]',
                              from_commit="a", to_commit="b", post=post)
    (c,) = post.calls
    assert c["url"] == f"{ub.URL}/api/claims/evaluate" and c["timeout"] == 1800
    assert c["json"] == {
        "question": "how does hello greet?",
        "claims": [{"name": "returns_greeting", "description": "The function returns a greeting string."},
                   {"name": "reviewed", "description": "Signed off.", "world": "closed"}],
        "rules": [{"head": "ready", "iff": {"and": ["returns_greeting", "reviewed"]}}],
        "from_commit": "a", "to_commit": "b"}
    assert set(out) == {"note", "id", "window", "judgments", "cost_usd"} and out["id"] == "abc"
    assert out["note"].startswith("Evaluated 2 conclusions over 2 passages.")
    assert out["judgments"]["returns_greeting"] == {
        "value": "true", "because": "evidence",
        "sources": [{"file": "pkg/app.py", "symbol": "hello", "line": 4,
                     "id": "pkg%2Fapp.py%3Ahello%3A4%3AFunctionDef#0-40"}]}
    assert "explanation" not in out["judgments"]["ready"]       # the compact view


def test_evaluate_claims_notes_for_no_passages_and_failures():
    note = ub._evaluate_claims("q", ["a: b"], post=_post_returning(_Resp(200, _claims_reply(J, passages=0))))["note"]
    assert note.startswith("No evidence") and "do not guess" in note
    note = ub._evaluate_claims("q", ["a: b"], post=_post_returning(_Resp(200, _claims_reply(J, failed=3, jobs=8))))["note"]
    assert note.startswith("3 of 8 judge calls failed")


def test_evaluate_claims_bad_input_and_errors_never_raise():
    post = _post_returning(_Resp(200, _claims_reply(J)))
    assert ub._evaluate_claims("q", [], post=post)["error"].startswith("bad input: give at least one claim")
    assert "name: description" in ub._evaluate_claims("q", ["no colon here"], post=post)["error"]
    assert "name: description" in ub._evaluate_claims("q", [": empty name"], post=post)["error"]
    assert ub._evaluate_claims("q", ["a: b"], rules="{not json", post=post)["error"].startswith("bad input")
    assert post.calls == []
    assert ub._evaluate_claims("q", ["a: b"], post=_post_returning(httpx.ConnectError("refused")))["error"].startswith(
        "unified-brain service not reachable")
    assert ub._evaluate_claims("q", ["a: b"], post=_post_returning(_Resp(400, {"detail": "claims: bad"})))["error"] == \
        "HTTP 400: claims: bad"
    assert "no evaluation" in ub._evaluate_claims("q", ["a: b"], post=_post_returning(_Resp(200, text="<html>")))["error"]
    assert "no evaluation" in ub._evaluate_claims("q", ["a: b"], post=_post_returning(_Resp(200, {"id": "x"})))["error"]


def test_evaluate_claims_per_symbol_sends_the_scope_and_compacts_rows():
    reply = _claims_reply({"calls_out": {"scope": "symbol", "summary": {"true": 1, "unknown": 1, "false": 0, "conflicted": 0},
                                         "by_symbol": [{"file": "a.py", "symbol": "f", "line": 1, "value": "true",
                                                        "sources": [], "explanation": {"big": "tree"}},
                                                       {"file": "a.py", "symbol": "g", "line": 9, "value": "unknown"}]}})
    post = _post_returning(_Resp(200, reply))
    out = ub._evaluate_claims("q", ["calls_out: The function calls another function."], per_symbol=True, post=post)
    assert post.calls[0]["json"]["claims"] == [{"name": "calls_out", "description": "The function calls another function.",
                                                 "scope": "symbol"}]
    assert out["judgments"]["calls_out"] == {
        "scope": "symbol", "summary": {"true": 1, "unknown": 1, "false": 0, "conflicted": 0},
        "by_symbol": [{"file": "a.py", "symbol": "f", "line": 1, "value": "true"},
                      {"file": "a.py", "symbol": "g", "line": 9, "value": "unknown"}]}
    # an explicit scope in a dict claim is kept
    post = _post_returning(_Resp(200, reply))
    ub._evaluate_claims("q", [{"name": "x", "description": "d", "scope": "evidence"}], per_symbol=True, post=post)
    assert post.calls[0]["json"]["claims"][0]["scope"] == "evidence"
