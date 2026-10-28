"""/api/claims/* against ingested_db with Jev stubbed: evaluation through the
question retrieval and through skill_ids, provenance back to file/symbol/line,
rules, conflicted evidence resolved by the KEEP/RETRACT pass, human
retraction, persistence, and the error paths."""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(service):
    with TestClient(service.app) as c:
        yield c


def _evaluate(client, **body):
    body.setdefault("question", "how does hello greet?")
    return client.post("/api/claims/evaluate", json=body)


CLAIMS = [
    {"name": "returns_greeting", "description": "The function returns a greeting string."},
    {"name": "returns_number", "description": "The function returns a numeric value."},
    {"name": "mentions_dragons", "description": "The code mentions dragons."},
    {"name": "reviewed", "description": "A reviewer signed the function off.", "world": "closed"},
]
RULES = [
    {"head": "ready", "iff": {"and": ["returns_greeting", {"not": "returns_number"}]}},
    {"head": "signed_off", "if": {"and": ["ready", "reviewed"]}},
]


def _arm(jev_mock):
    jev_mock.relevant = {"hello", "greet"}
    jev_mock.claims = {"returns_greeting": {"assert": ["hello, world"]},
                       "returns_number": {"deny": ["hello, world"]}}


def test_evaluate_through_retrieval(client, jev_mock):
    _arm(jev_mock)
    res = _evaluate(client, claims=CLAIMS, rules=RULES)
    assert res.status_code == 200, res.text
    d = res.json()
    values = {k: v["value"] for k, v in d["judgments"].items()}
    assert values == {"returns_greeting": "true", "returns_number": "false",
                      "mentions_dragons": "unknown", "reviewed": "false",
                      "ready": "true", "signed_off": "unknown"}
    # provenance: the greeting claim rests on hello's body
    (src,) = d["judgments"]["returns_greeting"]["sources"]
    assert (src["file"], src["symbol"], src["struct"]) == ("pkg/app.py", "hello", "FunctionDef")
    assert src["line"] >= 1 and src["id"].endswith("#0-" + str(src["id"].split("-")[-1]))
    assert src["id"] in {p["id"] for p in d["passages"]}
    assert d["judgments"]["reviewed"]["because"] == "closed-world absence"
    assert d["judgments"]["ready"]["sources"] == d["judgments"]["returns_greeting"]["sources"]
    tree = d["judgments"]["ready"]["explanation"]
    assert tree["claim"] == "ready() = true" and tree["given"][0]["because"].startswith("ready() iff and(")
    # the classification: 2 passages (hello, greet) x 4 claims, nothing failed
    cls = d["info"]["classification"]
    assert cls["passages"] == 2 and cls["jobs"] == 8 and cls["failed"] == 0 and cls["admitted"] == 2
    assert d["info"]["retrieval"]["mode"] == "question" and d["info"]["retrieval"]["evidence"] == 2
    calls = [c for c in jev_mock.calls if c["what"] == "claims"]
    assert len(calls) == 8 and all(len(c["state"]["candidates"]) == 1 for c in calls)
    assert all("pkg/app.py" not in json.dumps(c["state"]) for c in calls)   # only text travels
    assert d["cost_usd"] == pytest.approx(8 * 0.0001)
    # persisted and listed
    assert client.get(f"/api/claims/{d['id']}").json() == d
    listed = client.get("/api/claims").json()["evaluations"]
    assert listed[0]["id"] == d["id"] and listed[0]["values"]["ready"] == "true"
    assert client.get("/api/claims/nope").status_code == 404


def test_conflicting_passages_are_conflicted_until_resolved(client, jev_mock):
    jev_mock.relevant = {"hello", "greet"}
    # greet's body says `hello()`; hello's body says "hello, world": one passage
    # supports the claim, the other contradicts it
    jev_mock.claims = {"calls_hello": {"assert": ["return hello()"], "deny": ["hello, world"]}}
    claims = [{"name": "calls_hello", "description": "The code calls hello()."}]
    d = _evaluate(client, claims=claims).json()
    j = d["judgments"]["calls_hello"]
    assert j["value"] == "conflicted" and {s["symbol"] for s in j["sources"]} == {"hello", "greet"}
    assert j["because"] == "conflicting evidence"
    # the resolver retracts the passage Jev judges superseded
    jev_mock.standing = ["hello, world"]
    res = client.post(f"/api/claims/{d['id']}/resolve")
    assert res.status_code == 200, res.text
    r = res.json()
    assert r["judgments"]["calls_hello"]["value"] == "true"
    (step,) = r["resolution"]
    assert step["atom"] == "calls_hello()" and [x["source"] for x in step["retracted"]] == [
        next(s["id"] for s in j["sources"] if s["symbol"] == "hello")]
    assert len(step["kept"]) == 1 and r["history"][-1]["resolved"] == r["resolution"]
    assert [c["what"] for c in jev_mock.calls if c["what"] == "resolve"] == ["resolve"]
    assert len(r["kb"]["literals"]) == 1 and r["cost_usd"] == pytest.approx(d["cost_usd"] + 0.0001)
    # persisted
    assert client.get(f"/api/claims/{d['id']}").json()["judgments"]["calls_hello"]["value"] == "true"
    # nothing left to resolve: an empty pass is fine
    assert client.post(f"/api/claims/{d['id']}/resolve").json()["resolution"] == []


def test_retract_a_literal_by_hand(client, jev_mock):
    _arm(jev_mock)
    d = _evaluate(client, claims=CLAIMS[:2]).json()
    assert d["judgments"]["returns_number"]["value"] == "false"
    tree = d["judgments"]["returns_number"]["explanation"]
    (lit,) = tree["evidence"]
    assert lit["sign"] == "-" and lit["probability"] == 0.9
    res = client.post(f"/api/claims/{d['id']}/retract", json={"literal_ids": [lit["id"]]})
    assert res.status_code == 200, res.text
    r = res.json()
    assert r["judgments"]["returns_number"]["value"] == "unknown"
    assert r["judgments"]["returns_greeting"]["value"] == "true"
    assert r["history"][-1]["retracted"] == [lit["id"]]
    assert client.post(f"/api/claims/{d['id']}/retract", json={"literal_ids": [lit["id"]]}).status_code == 400


def test_evaluate_for_explicit_skill_ids_with_claim_params(client, ingested_db, jev_mock):
    r = ingested_db.r
    sid = next(s for s in r.smembers("skills_active")
               if json.loads(r.get(f"skill:{s}"))["def_key"].startswith("pkg/app.py:hello:"))
    jev_mock.claims = {"returns_value": {"assert": ["hello, world"]}}
    claims = [{"name": "returns_value", "description": "The function returns a value of the kind.",
               "params": {"kind": ["string", "number"]}}]
    rules = [{"head": "typed", "if": {"atom": {"predicate": "returns_value", "fields": {"kind": "string"}}}}]
    facts = [{"claim": "typed", "value": False, "source": "reviewer"}]
    d = _evaluate(client, question="", skill_ids=[sid], claims=claims, rules=rules, facts=facts).json()
    values = {k: v["value"] for k, v in d["judgments"].items()}
    # the stub answers by claim name, so both enumeration members are asserted;
    # the rule says typed, the reviewer's fact denies it: conflicted
    assert values == {"returns_value('string')": "true", "returns_value('number')": "true",
                      "typed": "conflicted"}
    assert d["info"]["retrieval"] == {"mode": "skill_ids", "evidence": 1}
    assert d["info"]["classification"]["passages"] == 1 and d["info"]["classification"]["jobs"] == 1
    (call,) = [c for c in jev_mock.calls if c["what"] == "claims"]
    assert [c["fields"] for c in call["state"]["candidates"]] == [{"kind": "string"}, {"kind": "number"}]
    assert d["passages"][0]["symbol"] == "hello" and d["window"] == ""
    assert "reviewer" in json.dumps(d["judgments"]["typed"]["explanation"])


def test_claims_errors(client, jev_mock, monkeypatch):
    _arm(jev_mock)
    one = [{"name": "returns_greeting", "description": "d"}]
    assert _evaluate(client, claims=[]).status_code == 400
    assert _evaluate(client, question="  ", claims=one).status_code == 400
    bad = [
        ([{"name": "bad name", "description": "d"}], [], [], "letters, digits"),
        (one + one, [], [], "listed twice"),
        ([{"name": "x", "description": " "}], [], [], "needs a description"),
        (one, [{"head": "h", "if": "returns_greeting", "iff": "returns_greeting"}], [], "exactly one"),
        (one, [{"head": "h", "if": "nope"}], [], "unknown claim"),
        (one, [{"head": "h", "if": {"xor": []}}], [], "operator"),
        (one, [], [{"claim": "nope"}], "names no claim"),
        ([{"name": "e", "description": "d", "params": {"k": []}}], [], [], "at least one member"),
    ]
    for claims, rules, facts, msg in bad:
        res = _evaluate(client, claims=claims, rules=rules, facts=facts)
        assert res.status_code == 400 and msg in res.json()["detail"], (msg, res.text)
    assert jev_mock.calls == []                                   # rejected before any Jev call
    res = _evaluate(client, question="", skill_ids=["no-such-skill"], claims=one)
    assert res.status_code == 400 and "no-such-skill" in res.json()["detail"]
    assert client.post("/api/claims/nope/retract", json={"literal_ids": ["L1"]}).status_code == 404
    assert client.post("/api/claims/nope/resolve").status_code == 404
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert "TYPESAFE_API_KEY" in _evaluate(client, claims=one).json()["detail"]
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    jev_mock.fail = "HTTP 401 Unauthorized"
    res = _evaluate(client, claims=one)
    assert res.status_code == 502 and res.json()["detail"].startswith("Jev scored none of")


def test_all_classifier_calls_failing_is_502(client, ingested_db, jev_mock):
    r = ingested_db.r
    sid = next(s for s in r.smembers("skills_active")
               if json.loads(r.get(f"skill:{s}"))["def_key"].startswith("pkg/app.py:hello:"))
    jev_mock.fail = "HTTP 503 down"
    res = _evaluate(client, question="", skill_ids=[sid],
                    claims=[{"name": "c", "description": "d"}])
    assert res.status_code == 502 and res.json()["detail"].startswith("Jev assessed none of 1 passages")


# ── per-symbol claims ───────────────────────────────────────────────────────

SYMBOL_CLAIMS = [
    {"name": "calls_out", "description": "The function calls another function.", "scope": "symbol"},
    {"name": "prints", "description": "The function prints to standard output.", "scope": "symbol"},
    {"name": "is_test", "description": "The function is a test.", "scope": "symbol", "world": "closed"},
]
NOISY = {"head": {"predicate": "noisy_caller", "fields": {"symbol": {"var": "S"}}},
         "iff": {"and": [{"atom": {"predicate": "calls_out", "fields": {"symbol": {"var": "S"}}}},
                         {"atom": {"predicate": "prints", "fields": {"symbol": {"var": "S"}}}}]}}


def _arm_symbols(jev_mock):
    jev_mock.relevant = {"hello", "greet", "Counter.inc", "helper"}
    jev_mock.claims = {"calls_out": {"assert": ["return hello()", "helper();"]},
                       "prints": {"assert": ["println"]}}


def test_per_symbol_claims_and_a_rule_with_a_variable(client, jev_mock):
    _arm_symbols(jev_mock)
    res = _evaluate(client, question="what calls what?", claims=SYMBOL_CLAIMS, rules=[NOISY])
    assert res.status_code == 200, res.text
    d = res.json()
    j = d["judgments"]
    assert set(j) == {"calls_out", "prints", "is_test", "noisy_caller"}
    assert all(v["scope"] == "symbol" for v in j.values())
    rows = {r["symbol"]: r for r in j["calls_out"]["by_symbol"]}
    assert {s: r["value"] for s, r in rows.items()} == {"hello": "unknown", "greet": "true",
                                                         "Counter.inc": "true", "helper": "unknown"}
    assert j["calls_out"]["summary"] == {"false": 0, "unknown": 2, "conflicted": 0, "true": 2}
    assert [(r["file"], r["line"]) for r in j["calls_out"]["by_symbol"]] == sorted(
        (r["file"], r["line"]) for r in j["calls_out"]["by_symbol"])
    # each symbol's answer rests on its own body only
    (src,) = rows["greet"]["sources"]
    assert src["symbol"] == "greet" and src["file"] == "pkg/app.py" and rows["greet"]["atom"].startswith("calls_out(symbol:pkg/app.py:greet:")
    assert {r["symbol"]: r["value"] for r in j["prints"]["by_symbol"]}["Counter.inc"] == "true"
    assert {r["value"] for r in j["is_test"]["by_symbol"]} == {"false"}          # closed world
    noisy = {r["symbol"]: r["value"] for r in j["noisy_caller"]["by_symbol"]}
    assert noisy == {"hello": "unknown", "greet": "unknown", "Counter.inc": "true", "helper": "unknown"}
    assert j["noisy_caller"]["by_symbol"][0]["explanation"]["claim"].startswith("noisy_caller(symbol:")
    # one Jev call per passage and claim, each with exactly its own symbol as the candidate
    calls = [c for c in jev_mock.calls if c["what"] == "claims"]
    cls = d["info"]["classification"]
    assert cls["passages"] == 4 and cls["jobs"] == len(calls) == 12 and cls["failed"] == 0
    for c in calls:
        (cand,) = c["state"]["candidates"]
        assert cand["fields"]["symbol"]["type"] == "symbol" and cand["fields"]["symbol"]["symbol"] in rows
    # persisted and rebuilt: a human retraction of Counter.inc's prints literal
    stored = client.get(f"/api/claims/{d['id']}").json()
    assert stored["judgments"] == j and stored["rules"] == [NOISY]
    inc = {r["symbol"]: r for r in j["prints"]["by_symbol"]}["Counter.inc"]
    (lit,) = inc["explanation"]["evidence"]
    r2 = client.post(f"/api/claims/{d['id']}/retract", json={"literal_ids": [lit["id"]]}).json()
    assert {r["symbol"]: r["value"] for r in r2["judgments"]["noisy_caller"]["by_symbol"]}["Counter.inc"] == "unknown"
    assert {r["symbol"]: r["value"] for r in r2["judgments"]["calls_out"]["by_symbol"]}["Counter.inc"] == "true"


def test_per_symbol_errors(client, jev_mock):
    _arm_symbols(jev_mock)
    one = [SYMBOL_CLAIMS[0]]
    bad = [
        (one, [{"head": {"predicate": "h", "fields": {"symbol": {"var": "Z"}}}, "if": "calls_out"}], "not used in the body"),
        (one, [{"head": {"predicate": "h", "fields": {"symbol": "literal"}}, "if": "calls_out"}], "variables only"),
        (one, [{"head": 7, "if": "calls_out"}], "a claim name or"),
        (one, [{"head": "h", "if": "calls_out"}], "fields are"),          # calls_out needs its symbol
        (one + [{"name": "yearly", "description": "d", "params": {"year": [2024]}}],
         [{"head": {"predicate": "h", "fields": {"x": {"var": "V"}}},
           "if": {"and": [{"atom": {"predicate": "yearly", "fields": {"year": {"var": "V"}}}},
                          {"atom": {"predicate": "calls_out", "fields": {"symbol": {"var": "V"}}}}]}}],
         "different types"),
        ([{"name": "c", "description": "d", "params": {"symbol": ["a"]}}], [], "per-symbol scope"),
    ]
    for claims, rules, msg in bad:
        res = _evaluate(client, claims=claims, rules=rules)
        assert res.status_code == 400 and msg in res.json()["detail"], (msg, res.text)
    assert jev_mock.calls == []


# ── call edges as facts ─────────────────────────────────────────────────────

REACHES = {"head": {"predicate": "reaches", "world": "closed",
                    "fields": {"src": {"var": "X"}, "dst": {"var": "Z"}}},
           "iff": {"or": [{"atom": {"predicate": "calls", "fields": {"caller": {"var": "X"}, "callee": {"var": "Z"}}}},
                          {"and": [{"atom": {"predicate": "calls", "fields": {"caller": {"var": "X"}, "callee": {"var": "Y"}}}},
                                   {"atom": {"predicate": "reaches", "fields": {"src": {"var": "Y"}, "dst": {"var": "Z"}}}}]}]}}
HAS_CALLEE = {"head": {"predicate": "has_callee", "fields": {"symbol": {"var": "S"}}, "world": "closed"},
              "iff": {"atom": {"predicate": "calls", "fields": {"caller": {"var": "S"}, "callee": {"var": "Y"}}}}}


def test_call_edges_among_the_fixture_symbols(ingested_db, service):
    import skills_lib
    r = ingested_db.r
    dks = sorted(json.loads(r.get(f"skill:{s}"))["def_key"] for s in r.smembers("skills_active"))
    edges = skills_lib.call_edges_among(r, dks)
    short = sorted((e["caller"].split(":")[1], e["callee"].split(":")[1], e["name"]) for e in edges)
    assert short == [("Counter.inc", "helper", "helper"), ("f", "area", "area"), ("greet", "hello", "hello")]
    assert all(e["line"] > 0 and len(e["commit"]) == 40 for e in edges)
    # a window at the first commit keeps only the python edge
    assert [e["name"] for e in skills_lib.call_edges_among(r, dks, to_rank=0)] == ["hello"]
    assert skills_lib.call_edges_among(r, ["not:a:def:key", "x"]) == []


def test_import_calls_and_follow_the_graph(client, jev_mock):
    jev_mock.relevant = {"hello", "greet", "Counter.inc", "helper"}
    jev_mock.claims = {"prints": {"assert": ["println"]}}
    claims = [{"name": "prints", "description": "The function prints.", "scope": "symbol"}]
    res = _evaluate(client, question="what calls what?", claims=claims, rules=[REACHES, HAS_CALLEE],
                    import_facts=["calls"])
    assert res.status_code == 200, res.text
    d = res.json()
    facts = d["info"]["facts"]
    assert facts["nodes"] == 4 and sorted(e["callee"].split(":")[1] for e in facts["calls"]) == ["hello", "helper"]
    j = d["judgments"]
    assert {r["symbol"]: r["value"] for r in j["has_callee"]["by_symbol"]} == {
        "hello": "false", "greet": "true", "Counter.inc": "true", "helper": "false"}
    reach = {(r["atom"]): r["value"] for r in j["reaches"]["by_symbol"]}
    by = {k.split("symbol:")[1].split(":")[1] + "->" + k.split("symbol:")[2].split(":")[1]: v for k, v in reach.items()}
    assert by["greet->hello"] == "true" and by["Counter.inc->helper"] == "true"
    # 4 x 4 minus the pairs naming one symbol twice, which candidates() drops (Factbag)
    assert by["greet->helper"] == "false" and by["hello->greet"] == "false" and len(by) == 12
    assert j["reaches"]["summary"] == {"false": 10, "unknown": 0, "conflicted": 0, "true": 2}
    # the facts are never classified: Jev saw one claim per passage, nothing about calls
    calls = [c for c in jev_mock.calls if c["what"] == "claims"]
    assert len(calls) == 4 and {c["questions"]["candidate_0"]["instructions"]["predicate"]["name"] for c in calls} == {"prints"}
    src = next(r for r in j["has_callee"]["by_symbol"] if r["symbol"] == "greet")["explanation"]
    assert "index:calls@" in json.dumps(src)
    assert d["import_facts"] == ["calls"] and client.get(f"/api/claims/{d['id']}").json()["judgments"] == j
    # errors
    assert "import_facts" in _evaluate(client, claims=claims, import_facts=["refs"]).json()["detail"]
    assert "index fact" in _evaluate(client, claims=[{"name": "calls", "description": "d"}],
                                     import_facts=["calls"]).json()["detail"]
    assert "world is open or closed" in _evaluate(
        client, claims=claims, rules=[{"head": {"predicate": "h", "world": "maybe"}, "if": "prints"}]).json()["detail"]


def test_qualified_symbols_find_their_callers(client, service, ingested_db):
    """Call sites name the bare method (`inc()`), so callers of `Counter.inc`
    live under called_by:inc; the lookups now follow every name."""
    r = ingested_db.r
    caller = "src/main.rs:main:1:function"
    r.sadd("called_by:inc", caller)
    r.hset(f"calls:{caller}", "inc:3", json.dumps({"snippet": "c.inc();", "commit": "deadbeef"}))
    try:
        d = client.get("/api/symbol/Counter.inc").json()
        assert [c["symbol"] for c in d["callers"]] == ["main"]
        text = service._callers_text("Counter.inc", r)
        assert "src/main.rs:main:1:function" in text and "c.inc();" in text
        assert "No callers" in service._callers_text("Counter", r)
    finally:
        r.srem("called_by:inc", caller)
        r.delete(f"calls:{caller}")
