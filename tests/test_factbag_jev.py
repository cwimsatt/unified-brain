"""factbag_jev with a scripted Jev: passage ids and splitting, the guard, the
classification request shape and admission rule (Factbag classifier.test.ts),
idempotent jobs, failures that write nothing, and the KEEP/RETRACT resolver
(Factbag resolve.test.ts)."""
import pytest

from factbag_core import Entity, KnowledgeBase, ProgrammerError, entity, enumeration, predicate, query
from factbag_jev import (CLAIM_CRITERIA, STANDING_CRITERIA, Guard, JevError, Passage,
                         UnsupportedPredicateError, admitted, candidates, classifier,
                         contradictions, parse_passage_id, passage_id, resolver, split_passages)


def dist(a, d, i):
    return {"ASSERT": a, "DENY": d, "IRRELEVANT": i}


def scripted(script):
    """A `post` that answers each call from `script` (a list of answers dicts
    or error strings) and records every request."""
    calls = []

    def post(client, state, questions, *, what, attempts=4):
        calls.append({"what": what, "state": state, "questions": questions})
        item = script.pop(0)
        if isinstance(item, str):
            return None, 0.0, item
        return item, 0.001, None
    post.calls = calls
    return post


def answers(*dists):
    return {f"candidate_{i}": {"choice": max(d, key=d.get), "probabilities": d}
            for i, d in enumerate(dists)}


@pytest.fixture
def claims():
    kb = KnowledgeBase()
    greets = predicate("returns_greeting", description="The function returns a greeting string.")
    numeric = predicate("returns_number", description="The function returns a numeric value.")
    kb._register(greets)
    kb._register(numeric)
    return kb, greets, numeric


P1 = Passage("pkg%2Fapp.py%3Ahello#0-30", "pkg/app.py:hello", 0, 30, "def hello():\n    return 'hi'",
             {"file": "pkg/app.py", "symbol": "hello", "line": 4})
P2 = Passage("pkg%2Fapp.py%3Ahello#30-60", "pkg/app.py:hello", 30, 60, "def hello():\n    return 42")


# ── passages ────────────────────────────────────────────────────────────────

def test_passage_ids_round_trip_and_reject_non_canonical_forms():
    pid = passage_id("src/lib.rs:Counter.inc@abc123", 10, 25)
    assert pid == "src%2Flib.rs%3ACounter.inc%40abc123#10-25"
    assert parse_passage_id(pid) == ("src/lib.rs:Counter.inc@abc123", 10, 25)
    for bad in ("doc", "doc#1", "doc#01-5", "doc#5-1", "a#b#0-1", "src/lib.rs#0-1", ""):
        with pytest.raises(ValueError):
            parse_passage_id(bad)


def test_split_passages_prefers_paragraphs_then_lines_then_words():
    text = "para one.\n\npara two is here.\nline three " * 3
    ps = split_passages("doc", text, size=40, overlap=5, meta={"file": "f"})
    assert [(p.start, p.end) for p in ps] == [(0, 11), (6, 29), (24, 51), (46, 69), (64, 91), (86, 120)]
    assert ps[0].text == "para one.\n\n" and ps[1].text.startswith("ne.\n\npara two is here.\n")
    assert all(p.doc_id == "doc" and p.meta == {"file": "f"} and p.id == passage_id("doc", p.start, p.end)
               for p in ps)
    assert split_passages("s", "short") == [Passage("s#0-5", "s", 0, 5, "short")]
    assert split_passages("e", "")[0].id == "e#0-0"
    hard = split_passages("h", "x" * 25, size=10, overlap=0)      # no separator: hard cuts
    assert [(p.start, p.end) for p in hard] == [(0, 10), (10, 20), (20, 25)]
    with pytest.raises(ProgrammerError):
        split_passages("d", "x", size=10, overlap=10)


# ── guard ───────────────────────────────────────────────────────────────────

def test_guard_is_deny_by_default_and_block_beats_permit(claims):
    _kb, greets, numeric = claims
    g = Guard()
    assert g.actions_for(greets) == set()
    g = g.permit("assert", greets, numeric).permit("deny", greets)
    assert g.actions_for(greets) == {"assert", "deny"} and g.actions_for(numeric) == {"assert"}
    assert g.block("deny", "*").actions_for(greets) == {"assert"}
    assert g.block("*", greets).actions_for(greets) == set() and g.names() == {"returns_greeting", "returns_number"}
    assert Guard().permit("*", "*").actions_for(numeric) == {"assert", "deny"}
    with pytest.raises(ProgrammerError):
        Guard().permit("query", greets)
    with pytest.raises(ProgrammerError):
        Guard().permit("assert")


# ── candidates ──────────────────────────────────────────────────────────────

def test_candidates_fan_out_over_finite_domains_and_skip_repeated_entities():
    kb = KnowledgeBase()
    a, b = Entity("sym", "a"), Entity("sym", "b")
    kb.entities.add(a, b)
    calls = predicate("calls", {"src": entity("sym"), "dst": entity("sym")}, description="src calls dst")
    assert [repr(x) for x in candidates(kb, calls)] == ["calls(sym:a, sym:b)", "calls(sym:b, sym:a)"]
    flag = predicate("flag", {"on": bool, "year": enumeration(2024, 2025)}, description="d")
    assert len(candidates(kb, flag)) == 4
    with pytest.raises(UnsupportedPredicateError, match="no description"):
        candidates(kb, predicate("nodesc"))
    with pytest.raises(UnsupportedPredicateError, match="open-ended"):
        candidates(kb, predicate("open", {"n": int}, description="d"))


# ── admission (Factbag jobs.ts admittedOutcomes) ────────────────────────────

@pytest.mark.parametrize("d,expected", [
    (dist(0.9, 0.05, 0.05), ["ASSERT"]),
    (dist(0.05, 0.9, 0.05), ["DENY"]),
    (dist(0.4, 0.3, 0.3), ["ASSERT"]),              # exactly the threshold
    (dist(0.39, 0.3, 0.31), []),                    # below it
    (dist(0.45, 0.1, 0.45), []),                    # IRRELEVANT ties the strongest
    (dist(0.5, 0.5, 0.0), ["ASSERT", "DENY"]),      # both: the KB reports conflicted
    (dist(0.55, 0.45, 0.0), ["ASSERT"]),            # margin 0: only the strongest
])
def test_admission_rule(d, expected):
    assert admitted(d, 0.4, 0.0) == expected


def test_admission_margin_admits_the_runner_up():
    assert admitted(dist(0.55, 0.45, 0.0), 0.4, 0.1) == ["ASSERT", "DENY"]
    assert admitted(dist(0.55, 0.45, 0.0), 0.5, 0.1) == ["ASSERT"]


# ── classification ──────────────────────────────────────────────────────────

def test_classifier_request_shape_is_factbags(claims):
    kb, greets, numeric = claims
    g = Guard().permit("assert", greets).permit("deny", greets).permit("assert", numeric)
    run = classifier([P1], kb, g)
    assert [(j.predicate.name, sorted(j.actions)) for j in run.jobs] == [
        ("returns_greeting", ["assert", "deny"]), ("returns_number", ["assert"])]
    state, questions = run.jobs[0].request()
    assert state == {"document": {"text": P1.text}, "candidates": [{"id": "candidate_0", "fields": {}}]}
    q = questions["candidate_0"]
    assert q["type"] == "choice" and q["criteria"] == CLAIM_CRITERIA
    assert q["instructions"]["task"].startswith("Classify the evidence in document.text for candidates[0].fields.")
    assert q["instructions"]["predicate"] == {
        "name": "returns_greeting", "metadata": {"description": "The function returns a greeting string."}}
    # ids, ranges and file names never travel to the model
    assert "pkg" not in str(state) and "hello#" not in str(questions)


def test_classifier_admits_evidence_with_source_and_probability(claims):
    kb, greets, numeric = claims
    g = Guard().permit("assert", greets).permit("deny", greets).permit("assert", numeric)
    run = classifier([P1], kb, g)
    post = scripted([answers(dist(0.9, 0.05, 0.05)), answers(dist(0.1, 0.8, 0.1))])
    results = run.execute_all(post)
    assert [e for _j, e in results] == [None, None] and [c["what"] for c in post.calls] == ["claims", "claims"]
    assert kb.literals() == []                       # the input is untouched
    (lit,) = run.kb.literals()
    assert (lit.sign, lit.source, lit.probability) == ("+", P1.id, 0.9)
    assert lit.metadata["jev"] == {"candidate": "candidate_0", "outcome": "ASSERT", "choice": "ASSERT",
                                   "confidence": None, "probabilities": dist(0.9, 0.05, 0.05)}
    # the DENY of returns_number was not permitted: assessed, not written
    assert run.jobs[1].result.candidates[0]["answer"]["choice"] == "DENY"
    assert run.jobs[1].result.candidates[0]["literals"] == []
    assert query(run.kb, greets()).value == "true" and query(run.kb, numeric()).value == "unknown"
    assert run.cost == pytest.approx(0.002)


def test_classifier_both_polarities_make_the_atom_conflicted(claims):
    kb, greets, _numeric = claims
    run = classifier([P1, P2], kb, Guard().permit("*", greets))
    post = scripted([answers(dist(0.9, 0.1, 0.0)), answers(dist(0.1, 0.85, 0.05))])
    run.execute_all(post)
    j = query(run.kb, greets())
    assert j.value == "conflicted" and {l.source for l in j.literals} == {P1.id, P2.id}


def test_classifier_batches_64_candidates_per_request():
    kb = KnowledgeBase()
    kb.entities.add(*[Entity("sym", f"s{i}") for i in range(70)])
    used = predicate("is_used", {"sym": entity("sym")}, description="The symbol is used.")
    kb._register(used)
    run = classifier([P1], kb)                        # bare kb: every predicate permitted
    assert [len(j.atoms) for j in run.jobs] == [64, 6]
    state, questions = run.jobs[0].request()
    assert len(state["candidates"]) == 64 and set(questions) == {f"candidate_{i}" for i in range(64)}
    assert state["candidates"][0]["fields"] == {"sym": {"type": "sym", "id": "s0"}}


def test_classifier_execute_is_idempotent_and_failures_write_nothing(claims):
    kb, greets, _numeric = claims
    run = classifier([P1], kb, Guard().permit("*", greets))
    job = run.jobs[0]
    post = scripted(["HTTP 503 down", answers({"ASSERT": 0.7, "DENY": 0.7, "IRRELEVANT": 0.0}),
                     answers(dist(0.9, 0.05, 0.05)), answers(dist(0.0, 1.0, 0.0))])
    with pytest.raises(JevError, match="503"):
        job.execute(post)
    assert job.state == "failed" and job.error == "HTTP 503 down" and run.kb.literals() == []
    with pytest.raises(JevError, match="distribution"):
        job.execute(post)                             # sums to 1.4: rejected before any write
    assert run.kb.literals() == [] and run.failures() == [job]
    first = job.execute(post)
    assert job.state == "done" and len(run.kb.literals()) == 1
    assert job.execute(post) is first and len(run.kb.literals()) == 1 and len(post.calls) == 3
    assert run.cost == pytest.approx(0.001)


def test_classifier_rejects_unknown_guard_names_and_bad_thresholds(claims):
    kb, greets, _numeric = claims
    other = predicate("other", description="d")
    with pytest.raises(ProgrammerError, match="does not have"):
        classifier([P1], kb, Guard().permit("assert", other))
    with pytest.raises(ProgrammerError):
        classifier([P1], kb, min_probability=0)
    with pytest.raises(UnsupportedPredicateError):
        kb._register(predicate("nodesc"))
        classifier([P1], kb)


# ── resolver (Factbag resolve.ts) ───────────────────────────────────────────

def test_resolver_retracts_superseded_passages_only(claims):
    kb, greets, _numeric = claims
    kb.asserts(greets(), source=P1.id, probability=0.9)
    kb.denies(greets(), source=P2.id, probability=0.8)
    assert len(contradictions(kb)) == 1
    run = resolver({P1.id: P1, P2.id: P2}, kb)
    (job,) = run.jobs
    state, questions = job.request()
    assert state == {"documents": [{"text": P1.text}, {"text": P2.text}], "candidate": {"fields": {}}}
    assert questions["passage_1"]["criteria"] == STANDING_CRITERIA
    assert "Judge documents[1] only" in questions["passage_1"]["instructions"]["task"]
    assert "not retracted for denying the fact" in questions["passage_1"]["instructions"]["task"]
    post = scripted([{"passage_0": {"choice": "KEEP", "probabilities": {"KEEP": 0.8, "RETRACT": 0.2}},
                      "passage_1": {"choice": "RETRACT", "probabilities": {"KEEP": 0.3, "RETRACT": 0.7}}}])
    (res,) = [r for _j, r in [(j, j.execute(post)) for j in run.jobs]]
    assert [r["source"] for r in res["retracted"]] == [P2.id] and [r["source"] for r in res["kept"]] == [P1.id]
    assert query(run.kb, greets()).value == "true" and query(kb, greets()).value == "conflicted"
    assert job.execute(post) is res and post.calls[0]["what"] == "resolve"


def test_resolver_thresholds_and_missing_sources(claims):
    kb, greets, _numeric = claims
    kb.asserts(greets(), source=P1.id)
    kb.denies(greets(), source=P2.id)
    run = resolver({P1.id: P1, P2.id: P2}, kb, min_probability=0.4, margin=0.3)
    post = scripted([{"passage_0": {"probabilities": {"KEEP": 0.45, "RETRACT": 0.55}},    # margin not met
                      "passage_1": {"probabilities": {"KEEP": 0.65, "RETRACT": 0.35}}}])  # below min
    res = run.jobs[0].execute(post)
    assert res["retracted"] == [] and len(res["kept"]) == 2 and len(run.kb.literals()) == 2
    kb.denies(greets(), source=None)
    with pytest.raises(ValueError, match="needs a passage source"):
        resolver({P1.id: P1, P2.id: P2}, kb)
    assert resolver({}, KnowledgeBase()).jobs == []


# ── candidate_filter (per-symbol claims) ─────────────────────────────────────

def test_candidate_filter_restricts_each_passage_to_its_own_symbol():
    kb = KnowledgeBase()
    a, b = Entity("symbol", "a.py:f:1:FunctionDef", file="a.py", symbol="f"), Entity("symbol", "a.py:g:5:FunctionDef")
    kb.entities.add(a, b)
    uses_io = predicate("uses_io", {"symbol": entity("symbol")}, description="The function does I/O.")
    kb.declare(uses_io)
    pa = Passage("x#0-1", a.id, 0, 1, "f", {})
    pb = Passage("y#0-1", b.id, 0, 1, "g", {})
    assert [len(j.atoms) for j in classifier([pa, pb], kb).jobs] == [2, 2]
    run = classifier([pa, pb], kb, candidate_filter=lambda p, atom: atom.fields["symbol"].id == p.doc_id)
    assert [(j.passage.id, [x.fields["symbol"].id for x in j.atoms]) for j in run.jobs] == [
        ("x#0-1", [a.id]), ("y#0-1", [b.id])]
    state, _q = run.jobs[0].request()
    assert state["candidates"] == [{"id": "candidate_0", "fields": {"symbol": {"type": "symbol", "id": a.id, "file": "a.py", "symbol": "f"}}}]
    # a passage with nothing left gets no job
    assert [j.passage.id for j in classifier([pa, pb], kb, candidate_filter=lambda p, atom: p is pa).jobs] == ["x#0-1"]
