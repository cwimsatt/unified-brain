"""Claims in the report probes: one probe of the fixture repository run
through _run_probe_fast with Jev stubbed and the LLM judge faked; the
findings cross-check in _persist_probe_result; reopen clearing; the page."""
import asyncio
import json

import pytest

import templates


class _Resp:
    class _U:
        prompt_tokens, completion_tokens = 10, 5

    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
        self.usage = self._U()


@pytest.fixture
def judge(service, monkeypatch):
    """litellm.acompletion -> a BENIGN verdict; records the calls."""
    calls = []

    async def fake(**kw):
        calls.append(kw)
        return _Resp("The change extends the greeting.\n\nVERDICT: BENIGN")
    monkeypatch.setattr(service.litellm, "acompletion", fake)
    return calls


def _hello_probe(service, ingested_db):
    """The probe unit for `hello` at the last commit (hello changed there)."""
    r = ingested_db.r
    commits = [c.decode() if isinstance(c, bytes) else c for c in r.zrange("commit_order", 0, -1)]
    items = service._expand_probe_units(service._collect_range_items(r, commits[0], commits[-1]))
    item = next(i for i in items if i["symbol"] == "hello" and i["commit"] == commits[-1])
    diffs = service._touch_diffs(r, item)
    q = service._build_probe_question(item, diffs, service._composite_member_delta(r, item),
                                      service._blob_consumption(r, item, None))
    ctx = service._probe_context(r, ingested_db.rb, q, item, None)
    return item, diffs, q, ctx


def test_probe_claims_are_judged_from_the_probe_texts(service, ingested_db, jev_mock, judge, monkeypatch):
    monkeypatch.setenv("PROBE_CLAIMS", "1")
    jev_mock.claims = {"introduces_defect": {"assert": ["hello, world"]},
                       "achieves_stated_goal": {"assert": ["hello, world"]},
                       "is_suspicious": {"deny": ["hello, world"]}}
    item, diffs, q, ctx = _hello_probe(service, ingested_db)
    assert any(d.get("diff") for d in diffs) and item["goal"].startswith("edit hello")
    ev, stats = {}, {"input": 0, "output": 0}
    text, verdict, err, findings = asyncio.run(service._run_probe_fast(
        q, ctx, item, diffs, service._llm_target(), ingested_db.r, ingested_db.rb, stats, record=ev))
    assert (verdict, err, findings) == ("BENIGN", "", []) and len(judge) == 1
    # the summary the probe record keeps
    summary = ev["claims_summary"]
    assert summary["values"] == {"achieves_stated_goal": "true", "introduces_defect": "true",
                                 "is_suspicious": "false", "is_dormant": "unknown"}
    assert sorted(summary["sources"]["introduces_defect"]) == ["body of hello", "diff of hello"]
    # the judgments the evidence record keeps: sources name the passage kind and coordinates
    cj = ev["claims"]
    assert {s["kind"] for s in cj["introduces_defect"]["sources"]} == {"body", "diff"}
    assert all(s["file"] == "pkg/app.py" and s["symbol"] == "hello" for s in cj["introduces_defect"]["sources"])
    assert cj["is_dormant"] == {"value": "unknown", "because": "no evidence", "sources": [], "evidence": []}
    assert cj["is_suspicious"]["evidence"][0]["sign"] == "-"
    run = ev["claims_run"]
    assert run["passages"] == 2 and run["jobs"] == 8 and run["failed"] == 0 and run["cost_usd"] == pytest.approx(0.0008)
    assert stats["jev_cost"] >= 0.0008                      # the pass is in the probe's Jev cost
    claims_calls = [c for c in jev_mock.calls if c["what"] == "claims"]
    assert len(claims_calls) == 8
    assert "edit hello" in claims_calls[0]["questions"]["candidate_0"]["instructions"]["predicate"]["metadata"]["description"] \
        or any("edit hello" in c["questions"]["candidate_0"]["instructions"]["predicate"]["metadata"]["description"]
               for c in claims_calls)
    # the judge still got the evidence text, not the claims
    assert "EVIDENCE GATHERED" in judge[0]["messages"][1]["content"]
    assert "[SYSTEM: one model call" in text


def test_probe_claims_can_be_switched_off(service, ingested_db, jev_mock, judge, monkeypatch):
    monkeypatch.setenv("PROBE_CLAIMS", "0")
    item, diffs, q, ctx = _hello_probe(service, ingested_db)
    ev = {}
    asyncio.run(service._run_probe_fast(q, ctx, item, diffs, service._llm_target(),
                                        ingested_db.r, ingested_db.rb, {"input": 0, "output": 0}, record=ev))
    assert "claims" not in ev and "claims_summary" not in ev
    assert not [c for c in jev_mock.calls if c["what"] == "claims"]


@pytest.fixture
def report(ingested_db):
    """A one-probe report skeleton under rid 'tclaims'; removed afterwards."""
    r = ingested_db.r
    rid, sid, c12 = "tclaims", "probe-1", "abcdef123456"
    rec = {"probe_id": sid, "skill_id": sid, "symbol": "hello", "loc": "pkg/app.py:4",
           "events": [], "commit": c12, "commits": [c12], "verdict": "PENDING", "answer": "", "error": ""}
    r.hset(f"report:{rid}:probes", sid, json.dumps(rec))
    r.hset(f"report:{rid}:vmap", sid, "PENDING")
    yield r, rid, sid, c12
    for k in list(r.scan_iter(match=f"report:{rid}*")):
        r.delete(k)


def _reopen(r, rid, sid):
    """What every reopen path does: back to PENDING, claims dropped."""
    rec = json.loads(r.hget(f"report:{rid}:probes", sid))
    rec.update({"verdict": "PENDING", "answer": "", "error": ""})
    rec.pop("claims", None)
    r.hset(f"report:{rid}:probes", sid, json.dumps(rec))
    r.hset(f"report:{rid}:vmap", sid, "PENDING")


def test_harsher_claims_write_a_finding_and_milder_ones_a_note(service, report, monkeypatch):
    r, rid, sid, c12 = report
    monkeypatch.setenv("PROBE_CLAIMS_FINDINGS", "1")
    claims = {"values": {"achieves_stated_goal": "true", "introduces_defect": "true",
                         "is_suspicious": "unknown", "is_dormant": "unknown"},
              "sources": {"introduces_defect": ["body of hello", "diff of hello"]}}
    assert service._persist_probe_result(r, rid, sid, "BENIGN", "VERDICT: BENIGN", "", [], "m",
                                         extra={"claims": claims}) is True
    rec = json.loads(r.hget(f"report:{rid}:probes", sid))
    assert rec["verdict"] == "BENIGN" and rec["claims"]["disagree"] == "harsher"
    assert rec["claims"]["findings"] == ["introduces_defect"] and rec["claims"]["values"]["introduces_defect"] == "true"
    key = f"pkg/app.py@{c12}|claims:{sid}"
    f = json.loads(r.hget(f"report:{rid}:findings", key))
    assert f["severity"] == "MAJOR_FLAW" and f["by"] == f"claims:{sid}" and f["model"] == service.JEV_MODEL
    assert "introduces_defect is true" in f["note"] and "body of hello" in f["note"]
    assert service._SEV_RANK["MAJOR_FLAW"] > service._SEV_RANK["BENIGN"]      # what makes it contested
    # a finding by "claims:<sid>" is a lead for the probe's own rerun
    leads = service._leads_for(r, rid, {"loc": "pkg/app.py:4", "commit": c12}, sid)
    assert [l["severity"] for l in leads] == ["MAJOR_FLAW"]
    # the judge is harsher than the claims: a note, no finding
    _reopen(r, rid, sid)
    milder = {"values": {"achieves_stated_goal": "true", "introduces_defect": "false",
                         "is_suspicious": "false", "is_dormant": "unknown"}, "sources": {}}
    assert service._persist_probe_result(r, rid, sid, "MAJOR_FLAW", "VERDICT: MAJOR_FLAW", "", [], "m",
                                         extra={"claims": milder})
    rec = json.loads(r.hget(f"report:{rid}:probes", sid))
    assert rec["claims"]["disagree"] == "milder" and rec["claims"]["findings"] == []
    assert len(r.hgetall(f"report:{rid}:findings") or {}) == 1
    # agreement: nothing to say
    _reopen(r, rid, sid)
    service._persist_probe_result(r, rid, sid, "BENIGN", "", "", [], "m", extra={"claims": milder})
    assert json.loads(r.hget(f"report:{rid}:probes", sid))["claims"]["disagree"] is None
    # findings switched off: the harsher claim is recorded, no finding is written
    _reopen(r, rid, sid)
    r.hdel(f"report:{rid}:findings", key)
    monkeypatch.setenv("PROBE_CLAIMS_FINDINGS", "0")
    service._persist_probe_result(r, rid, sid, "BENIGN", "", "", [], "m", extra={"claims": claims})
    rec = json.loads(r.hget(f"report:{rid}:probes", sid))
    assert rec["claims"]["findings"] == [] and rec["claims"]["disagree"] is None
    assert len(r.hgetall(f"report:{rid}:findings") or {}) == 0
    # a run without claims (PROBE_CLAIMS=0) leaves no claims key behind
    _reopen(r, rid, sid)
    service._persist_probe_result(r, rid, sid, "BENIGN", "", "", [], "m", extra={"claims": None})
    assert "claims" not in json.loads(r.hget(f"report:{rid}:probes", sid))


def test_reopening_a_probe_clears_its_claims(service, report):
    r, rid, sid, _c12 = report
    claims = {"values": {"achieves_stated_goal": "true", "introduces_defect": "false",
                         "is_suspicious": "false", "is_dormant": "false"}, "sources": {}}
    service._persist_probe_result(r, rid, sid, "BENIGN", "", "", [], "m", extra={"claims": claims})
    assert "claims" in json.loads(r.hget(f"report:{rid}:probes", sid))
    service._rewind_probe(r, rid, sid)
    rec = json.loads(r.hget(f"report:{rid}:probes", sid))
    assert "claims" not in rec and rec["verdict"] == "PENDING"


def test_timeline_page_renders_claim_badges_and_the_evidence_section():
    h = templates.TIMELINE_HTML
    assert "claimBadges(p.claims)" in h and "head('Claims (Jev)')" in h
    assert ".badge.claim.bad" in h and "j.kind === 'jev'" not in h
    assert "(p.claims || {}).values" in h            # part of the redraw fingerprint
