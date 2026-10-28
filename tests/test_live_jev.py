"""Opt-in: the claims layer against the real Jev. Skipped unless
UB_LIVE_TESTS=1 and a Jev key is present (TYPESAFE_API_KEY, OPENROUTER_API_KEY
or JEV_API_KEY). Costs a few calls; records nothing."""
import os

import pytest

import jev_client
from factbag_core import KnowledgeBase, predicate, query
from factbag_jev import Guard, classifier, resolver, split_passages

pytestmark = pytest.mark.skipif(
    os.environ.get("UB_LIVE_TESTS") != "1"
    or not any(os.environ.get(k) for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "JEV_API_KEY")),
    reason="live Jev: set UB_LIVE_TESTS=1 and a Jev key")

HELLO = 'def hello():\n    """Greet the caller."""\n    return "hello, world"\n'


def test_jev_classifies_a_function_body_against_claims():
    kb = KnowledgeBase()
    greets = predicate("returns_greeting", description="The function returns a greeting string.")
    numeric = predicate("returns_number", description="The function returns a numeric value such as an integer or float.")
    dragons = predicate("mentions_dragons", description="The code mentions dragons.")
    kb.declare(greets, numeric, dragons)
    passages = split_passages("pkg/app.py:hello:1:FunctionDef", HELLO)
    run = classifier(passages, kb, Guard().permit("*", greets, numeric, dragons))
    client = jev_client._jev_client()
    try:
        results = run.execute_all(jev_client._jev_post, client)
    finally:
        client.close()
    assert [e for _j, e in results] == [None, None, None], results
    values = {p.name: query(run.kb, p()).value for p in (greets, numeric, dragons)}
    answers = {j.predicate.name: j.result.candidates[0]["answer"]["probabilities"] for j in run.jobs}
    print("\nlive Jev:", values, answers, f"${run.cost:.5f}")
    assert values["returns_greeting"] == "true"
    assert values["returns_number"] in ("false", "unknown")
    assert values["mentions_dragons"] == "unknown"
    assert run.cost > 0
    for p in answers.values():
        assert abs(sum(p.values()) - 1) < 0.05


def test_jev_resolves_a_superseded_passage():
    kb = KnowledgeBase()
    limit = predicate("rate_limit_is_100", description="The documented request rate limit is 100 requests per minute.")
    kb.declare(limit)
    old = split_passages("docs/v1", "API guide, January 2024.\nRate limit: 100 requests per minute per key.")[0]
    new = split_passages("docs/v2", "API guide, March 2025 (supersedes January 2024).\nThe rate limit "
                                    "was raised: it is now 500 requests per minute per key, not 100.")[0]
    kb.asserts(limit(), source=old.id, probability=0.9)
    kb.denies(limit(), source=new.id, probability=0.9)
    assert query(kb, limit()).value == "conflicted"
    run = resolver({old.id: old, new.id: new}, kb)
    client = jev_client._jev_client()
    try:
        results = run.execute_all(jev_client._jev_post, client)
    finally:
        client.close()
    assert results[0][1] is None, results
    res = run.jobs[0].result
    print("\nlive resolver:", [(r["source"], r["answer"]["probabilities"]) for r in res["retracted"] + res["kept"]])
    # The superseded January passage goes, the March one stays, so the claim
    # is false. (With Factbag's original wording Jev retracted both on
    # 2026-10-28; see scripts/claims_resolver_eval.py.)
    assert [r["source"] for r in res["retracted"]] == [old.id]
    assert query(run.kb, limit()).value == "false"
    assert run.cost > 0


def test_live_probe_claims_on_the_fixture_change(service, ingested_db):
    """The four probe claims, judged by the real Jev from hello's body and
    the diff that changed its greeting. Prints the values and cost."""
    r = ingested_db.r
    commits = [c.decode() if isinstance(c, bytes) else c for c in r.zrange("commit_order", 0, -1)]
    items = service._expand_probe_units(service._collect_range_items(r, commits[0], commits[-1]))
    item = next(i for i in items if i["symbol"] == "hello" and i["commit"] == commits[-1])
    diffs = service._touch_diffs(r, item)
    body = 'def hello():\n    return "hello, world"\n'
    passages = service._probe_claim_passages("pkg/app.py:hello:4:FunctionDef", "pkg/app.py", "hello",
                                             item, body, diffs, [])
    info = {"jev_cost": 0.0}
    res = service._probe_claims(item["goal"], passages, info)
    print("\nlive probe claims:", res["summary"]["values"], f"${res['cost_usd']:.5f}",
          {k: v["evidence"][:1] for k, v in res["judgments"].items() if v["evidence"]},
          {k: res[k] for k in ("passages", "jobs", "failed", "first_error")})
    # mechanics only: Jev's reading of a greeting edit against a commit
    # message that also names other files is its own (on 2026-10-28 it denied
    # achieves_stated_goal at 0.54 from the body alone)
    assert res["failed"] == 0, res["first_error"]
    assert res["jobs"] == 4 * res["passages"] and res["passages"] >= 2
    assert res["cost_usd"] > 0 and info["jev_cost"] == pytest.approx(res["cost_usd"], abs=1e-6)
    assert res["summary"]["values"]["is_suspicious"] != "true"
