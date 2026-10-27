"""The service against ingested_db with Jev stubbed: /api/retrieve (the
pre-filter modes, the all-failed 502), /api/grep (400, cap), /health, the
Jev provider table, cost and the retrying _jev_post helper."""
import contextvars
import json

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(service):
    with TestClient(service.app) as c:
        yield c


def _retrieve(client, question="where is hello defined?"):
    return client.post("/api/retrieve", json={"question": question})


def _symbols_scored(jev_mock):
    return {c["state"].get("symbol") for c in jev_mock.calls if c["what"] == "select"}


# ── /api/retrieve ────────────────────────────────────────────────────────────

def test_retrieve_prefilter_none_scores_every_skill(client, jev_mock):
    jev_mock.relevant = {"hello"}
    res = _retrieve(client)
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["info"]["prefilter"] == {"store": "none", "ids": 0, "fallback": False}
    assert any("Pre-filter off" in line for line in d["log"])
    sel = d["info"]["select"]
    assert sel["failed"] == 0 and sel["kept"] == 1 and sel["scored"] == sel["candidates"]
    assert sel["candidates"] == len(jev_mock.calls) > 5
    assert sel["first_error"] is None
    assert {"hello", "greet", "Counter.inc", "helper", "box", "<module>"} <= _symbols_scored(jev_mock)
    (row,) = d["evidence"]
    assert (row["file"], row["symbol"], row["struct"]) == ("pkg/app.py", "hello", "FunctionDef")
    assert row["line"] >= 1 and row["how"] == "scored by Jev" and row["score"] == 4.0
    assert "def hello" in row["code"] and 'return "hello, world"' in row["code"]
    assert d["leads"] == [] and d["window"] == "" and d["question"] == "where is hello defined?"
    assert "expand" not in d["info"]                  # JEV_EXPAND=0


def test_retrieve_expand_runs_the_chain_through_the_stub(client, jev_mock, monkeypatch):
    """JEV_EXPAND=1 (the production default): the name-following pass runs
    its verdict + names question through _jev_post as one `chain` call per
    seed, with the options under `criteria`; every seed is rejected by the
    stub, so no lead is added and info.expand is still populated."""
    monkeypatch.setenv("JEV_EXPAND", "1")
    jev_mock.relevant = {"hello", "greet"}
    res = _retrieve(client)
    assert res.status_code == 200, res.text
    d = res.json()
    assert {e["symbol"] for e in d["evidence"]} == {"hello", "greet"}
    x = d["info"]["expand"]
    assert x["hops"] >= 1 and x["scored"] == 2 and x["added"] == 0
    assert any(m.startswith("Expansion added 0 leads") for m in d["log"])
    assert not any(m.startswith("Expansion failed") for m in d["log"])
    chain = [c for c in jev_mock.calls if c["what"] == "chain"]
    # one call per candidate the greps for the seeds' names turned up (the
    # state is the candidate, not the seed): Greeter and the module body
    assert len(chain) == x["scored"] == 2
    assert {c["state"].get("symbol") for c in chain} == {"Greeter", "<module>"}
    for c in chain:
        v = c["questions"]["v"]
        assert v["type"] == "choice" and set(v["criteria"]) == {"accept", "continue", "reject"}
        assert c["questions"]["names"]["type"] == "choice"
    assert {c["what"] for c in jev_mock.calls} == {"select", "chain"}


def test_retrieve_no_relevant_body_is_empty_evidence(client, jev_mock):
    jev_mock.relevant = set()
    res = _retrieve(client)
    assert res.status_code == 200
    d = res.json()
    assert d["evidence"] == []
    assert d["info"]["select"]["scored"] == d["info"]["select"]["candidates"] > 0
    assert d["info"]["select"]["kept"] == 0


def _with_store(service, monkeypatch, name, knn):
    monkeypatch.setattr(service, "_VECTOR_STORE",
                        contextvars.ContextVar("vector_store", default=name))
    monkeypatch.setattr(service, "_knn_candidates", knn)


def test_retrieve_knn_empty_falls_back_to_every_skill(client, service, jev_mock, monkeypatch):
    jev_mock.relevant = {"hello"}
    baseline = _retrieve(client).json()["info"]["select"]["candidates"]
    jev_mock.calls.clear()
    _with_store(service, monkeypatch, "voyage", lambda *a, **kw: [])
    d = _retrieve(client).json()
    assert d["info"]["prefilter"] == {"store": "voyage", "ids": 0, "fallback": True}
    assert d["info"]["select"]["candidates"] == baseline == len(jev_mock.calls)
    assert any("--embed-only" in line for line in d["log"])
    assert [r["symbol"] for r in d["evidence"]] == ["hello"]


def test_retrieve_knn_error_falls_back_to_every_skill(client, service, jev_mock, monkeypatch):
    jev_mock.relevant = {"hello"}
    baseline = _retrieve(client).json()["info"]["select"]["candidates"]

    def boom(*a, **kw):
        raise RuntimeError("vec table missing")
    _with_store(service, monkeypatch, "qwen", boom)
    d = _retrieve(client).json()
    assert d["info"]["prefilter"] == {"store": "qwen", "ids": 0, "fallback": True}
    assert d["info"]["select"]["candidates"] == baseline
    assert any("Embedding search failed" in line for line in d["log"])
    assert [r["symbol"] for r in d["evidence"]] == ["hello"]


def test_retrieve_knn_ids_limit_the_candidates(client, service, ingested_db, jev_mock, monkeypatch):
    r = ingested_db.r
    sid = next(s for s in r.smembers("skills_active")
               if json.loads(r.get(f"skill:{s}"))["def_key"].startswith("pkg/app.py:hello:"))
    jev_mock.relevant = {"hello"}
    _with_store(service, monkeypatch, "voyage", lambda *a, **kw: [(sid, 0.9)])
    d = _retrieve(client).json()
    assert d["info"]["prefilter"] == {"store": "voyage", "ids": 1, "fallback": False}
    assert d["info"]["select"]["candidates"] == 1 == len(jev_mock.calls)
    assert [r_["symbol"] for r_ in d["evidence"]] == ["hello"]
    assert d["evidence"][0]["skill_id"] == sid
    assert d["info"]["seed_ranks"] == [1]


def test_retrieve_all_jev_calls_failed_is_502(client, jev_mock):
    jev_mock.fail = "HTTP 401 Unauthorized"
    res = _retrieve(client)
    assert res.status_code == 502, res.text
    detail = res.json()["detail"]
    assert detail.startswith("Jev scored none of") and "HTTP 401 Unauthorized" in detail
    assert len(jev_mock.calls) > 1


def test_retrieve_empty_question_is_400(client, jev_mock):
    assert _retrieve(client, "   ").status_code == 400
    assert jev_mock.calls == []


def test_retrieve_without_jev_key_is_400(client, jev_mock, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    res = _retrieve(client)
    assert res.status_code == 400
    assert "TYPESAFE_API_KEY" in res.json()["detail"]
    assert jev_mock.calls == []


# ── /api/grep, /health ───────────────────────────────────────────────────────

def test_grep_finds_bodies(client):
    d = client.get("/api/grep", params={"pattern": "hello, world"}).json()
    assert d["pattern"] == "hello, world" and d["total_matched"] == 1
    (row,) = d["rows"]
    assert (row["file"], row["symbol"], row["struct"]) == ("pkg/app.py", "hello", "FunctionDef")
    assert row["hit_line"] >= row["line"] and "hello, world" in row["snippet"]
    d = client.get("/api/grep", params={"pattern": "fn inc", "file": "lib.rs"}).json()
    assert {r["symbol"] for r in d["rows"]} == {"Counter", "Counter.inc"}


def test_grep_regex(client):
    d = client.get("/api/grep", params={"pattern": r"helper\(\)", "regex": "true"}).json()
    # the call site (in the impl and its method) and `fn helper() {}` itself
    assert {r["symbol"] for r in d["rows"]} == {"Counter", "Counter.inc", "helper"}
    d = client.get("/api/grep", params={"pattern": r"^fn helper", "regex": "true"}).json()
    assert [r["symbol"] for r in d["rows"]] == ["helper"]


def test_grep_bad_regex_is_400(client):
    res = client.get("/api/grep", params={"pattern": "(unclosed", "regex": "true"})
    assert res.status_code == 400
    assert res.json()["detail"].startswith("invalid regex:")
    # the same text is a fine literal
    assert client.get("/api/grep", params={"pattern": "(unclosed"}).status_code == 200


def test_grep_empty_pattern_is_400(client):
    assert client.get("/api/grep", params={"pattern": ""}).status_code == 400


def test_grep_max_results_is_capped(client, service, monkeypatch):
    assert service.GREP_MAX_RESULTS == 500
    monkeypatch.setattr(service, "GREP_MAX_RESULTS", 1)
    d = client.get("/api/grep", params={"pattern": "def ", "max_results": 10000}).json()
    assert len(d["rows"]) == 1 and d["total_matched"] >= 3
    d = client.get("/api/grep", params={"pattern": "def ", "max_results": 0}).json()
    assert len(d["rows"]) == 1


def test_health(client, ingested_db):
    assert client.get("/health").json() == {"status": "ok", "db": ingested_db.path}


# ── Jev provider table ───────────────────────────────────────────────────────

@pytest.fixture
def clean_jev_env(monkeypatch):
    for k in ("JEV_PROVIDER", "JEV_MODEL", "JEV_API_BASE", "JEV_API_KEY",
              "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_provider_openrouter_only_key_is_upstream(service, clean_jev_env):
    clean_jev_env.setenv("OPENROUTER_API_KEY", "or-key")
    p = service._jev_provider()
    assert p["name"] == "openrouter"
    assert p["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert p["model"] == "typesafe/jev-1.13" and p["key"] == "or-key"
    assert p["key_env"] == "OPENROUTER_API_KEY"
    assert service._jev_key_missing() == ""


def test_provider_typesafe_only_key(service, clean_jev_env):
    clean_jev_env.setenv("TYPESAFE_API_KEY", "ts-key")
    p = service._jev_provider()
    assert p["name"] == "typesafe" and p["url"] == "https://api.typesafe.ai/v1/systemone"
    assert p["model"] == "jev-1.13.0" and p["key"] == "ts-key"


def test_provider_both_keys_defaults_to_openrouter(service, clean_jev_env):
    clean_jev_env.setenv("TYPESAFE_API_KEY", "ts")
    clean_jev_env.setenv("OPENROUTER_API_KEY", "or")
    assert service._jev_provider()["name"] == "openrouter"
    clean_jev_env.setenv("JEV_PROVIDER", "typesafe")
    assert service._jev_provider()["name"] == "typesafe"


def test_provider_no_key_names_the_variable(service, clean_jev_env):
    clean_jev_env.setenv("JEV_PROVIDER", "typesafe")
    msg = service._jev_key_missing()
    assert "TYPESAFE_API_KEY" in msg and "JEV_API_KEY" in msg
    assert service._jev_provider()["key"] == ""


def test_provider_overrides_and_aliases(service, clean_jev_env):
    clean_jev_env.setenv("JEV_PROVIDER", "typesafe")
    clean_jev_env.setenv("JEV_API_KEY", "override")
    clean_jev_env.setenv("TYPESAFE_API_KEY", "ignored")
    clean_jev_env.setenv("JEV_API_BASE", "http://127.0.0.1:9/v1")
    clean_jev_env.setenv("JEV_MODEL", "typesafe/jev-1.13")       # the openrouter name
    p = service._jev_provider()
    assert (p["key"], p["url"], p["model"]) == ("override", "http://127.0.0.1:9/v1", "jev-1.13.0")
    clean_jev_env.setenv("JEV_PROVIDER", "openrouter")
    clean_jev_env.setenv("JEV_MODEL", "jev-1.13.0")
    assert service._jev_provider()["model"] == "typesafe/jev-1.13"
    clean_jev_env.setenv("JEV_MODEL", "jev-latest")
    assert service._jev_provider()["model"] == "jev-latest"
    assert service._jev_client().headers["Authorization"] == "Bearer override"


def test_provider_unknown_raises(service, clean_jev_env):
    clean_jev_env.setenv("JEV_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="JEV_PROVIDER"):
        service._jev_provider()


def test_jev_cost(service, monkeypatch):
    assert service._jev_cost({"usage": {"cost": 0.00123, "input_tokens": 999}}) == 0.00123
    assert service._jev_cost({"usage": {"cost": 0, "input_tokens": 999}}) == 0.0
    monkeypatch.setenv("JEV_PRICE_PER_MTOK_IN", "0.042")
    assert service._jev_cost({"usage": {"input_tokens": 1_000_000}}) == pytest.approx(0.042)
    monkeypatch.setenv("JEV_PRICE_PER_MTOK_IN", "1.0")
    assert service._jev_cost({"usage": {"input_tokens": 500_000}}) == pytest.approx(0.5)
    assert service._jev_cost({}) == 0.0


# ── _jev_post ────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status, body=None, text=None, headers=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class _Client:
    """A fake httpx.Client: answers from a script, records every post."""
    def __init__(self, script):
        self.script = list(script)
        self.posts = []

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep(service, monkeypatch):
    slept = []
    monkeypatch.setattr(service.time, "sleep", lambda s: slept.append(s))
    return slept


OK = {"answers": {"rel": {"score": 3}}, "usage": {"input_tokens": 1000}}


def test_jev_post_success_adds_model_and_prices_tokens(service, monkeypatch, no_sleep):
    monkeypatch.setenv("JEV_PRICE_PER_MTOK_IN", "0.042")
    c = _Client([_Resp(200, OK)])
    answers, cost, err = service._jev_post(c, {"code": "x"}, {"rel": {"type": "score"}}, what="t")
    assert (answers, err) == ({"rel": {"score": 3}}, None)
    assert cost == pytest.approx(0.042 * 1000 / 1e6)
    (p,) = c.posts
    assert p["url"] == service._jev_provider()["url"]
    assert p["json"] == {"model": service._jev_provider()["model"],
                         "state": {"code": "x"}, "questions": {"rel": {"type": "score"}}}
    assert p["headers"] is None and no_sleep == []


def test_jev_post_retries_429_with_retry_after(service, no_sleep, capsys):
    c = _Client([_Resp(429, text="slow down", headers={"Retry-After": "7"}),
                 _Resp(503, text="down"),
                 _Resp(200, OK)])
    answers, _cost, err = service._jev_post(c, {}, {}, what="select")
    assert err is None and answers == OK["answers"]
    assert no_sleep == [7, 2]                        # Retry-After, then 2**1
    assert len(c.posts) == 3
    out = capsys.readouterr().out
    assert out.count("retrying") == 1 and "HTTP 429" in out


def test_jev_post_retry_after_is_capped(service, no_sleep):
    c = _Client([_Resp(429, headers={"Retry-After": "600"}), _Resp(200, OK)])
    assert service._jev_post(c, {}, {}, what="t")[2] is None
    assert no_sleep == [60]


def test_jev_post_gives_up_without_a_final_sleep(service, no_sleep):
    c = _Client([_Resp(529, text="overloaded")] * 4)
    answers, cost, err = service._jev_post(c, {}, {}, what="t")
    assert answers is None and cost == 0.0 and err.startswith("HTTP 529")
    assert len(c.posts) == 4 and len(no_sleep) == 3    # no sleep after the last attempt
    c = _Client([httpx.ConnectError("refused")] * 2)
    answers, cost, err = service._jev_post(c, {}, {}, what="t", attempts=2)
    assert answers is None and "ConnectError" in err and no_sleep[3:] == [1]


def test_jev_post_non_retryable_errors(service, no_sleep):
    answers, _c, err = service._jev_post(_Client([_Resp(401, {"error": "bad key"})]), {}, {}, what="t")
    assert answers is None and err.startswith("HTTP 401")
    answers, _c, err = service._jev_post(_Client([_Resp(200, text="<html>")]), {}, {}, what="t")
    assert answers is None and "not JSON" in err
    answers, _c, err = service._jev_post(_Client([_Resp(200, ["a", "list"])]), {}, {}, what="t")
    assert answers is None and err.startswith("HTTP 200")
    assert no_sleep == []


def test_jev_post_long_context_retry_on_typesafe(service, monkeypatch, no_sleep):
    monkeypatch.setenv("JEV_PROVIDER", "typesafe")
    c = _Client([_Resp(400, {"error": {"code": "max_tokens_exceeded"}}), _Resp(200, OK)])
    answers, _cost, err = service._jev_post(c, {}, {}, what="t", attempts=1)
    assert err is None and answers == OK["answers"]
    assert c.posts[0]["headers"] is None
    assert c.posts[1]["headers"] == {"TypeSafe-Danger-Zone-Long-Context": "true"}
    # second 400 of the same kind is final
    c = _Client([_Resp(400, {"error": "max_tokens_exceeded"})] * 2)
    assert service._jev_post(c, {}, {}, what="t")[2].startswith("HTTP 400")
    assert len(c.posts) == 2
    # openrouter: a 400 is final at once
    monkeypatch.setenv("JEV_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    c = _Client([_Resp(400, {"error": "max_tokens_exceeded"})])
    assert service._jev_post(c, {}, {}, what="t")[2].startswith("HTTP 400")
    assert len(c.posts) == 1 and no_sleep == []


# ── /api/symbol ──────────────────────────────────────────────────────────────

def test_api_symbol_resolves_a_method_by_its_bare_name(client):
    d = client.get("/api/symbol/inc").json()
    assert [x["symbol"] for x in d["definitions"]] == ["Counter.inc"]
    assert d["definitions"][0]["file"] == "src/lib.rs" and d["definitions"][0]["has_body"]
    full = client.get("/api/symbol/Counter.inc").json()
    assert [x["symbol"] for x in full["definitions"]] == ["Counter.inc"]
    assert {c["symbol"] for c in client.get("/api/symbol/helper").json()["callers"]} == {"Counter.inc"}
