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
