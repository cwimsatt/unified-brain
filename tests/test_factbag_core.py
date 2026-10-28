"""factbag_core: the four-valued semantics, ported case by case from Factbag's
declarative test cases (packages/core/examples/test-cases) and its
docs/formalized notes; plus persistence, fork isolation and the DSL errors."""
import datetime as dt
import itertools
import random

import pytest

from factbag_core import (And, Entity, KnowledgeBase, ProgrammerError, and_, closed_predicate,
                          entity, enumeration, explain, formula_from_dict, formula_to_dict,
                          negate, negate_as_failure, not_, or_, predicate, query, variable,
                          vmax, vmin)

T, F, U, C = "true", "false", "unknown", "conflicted"


@pytest.fixture
def people():
    return {n: Entity("person", n) for n in ("alice", "bob", "cara", "drew")}


# ── values (docs/formalized/values.md, operators.md, negation.md) ─────────────

def test_lattice_min_max_and_negations():
    assert vmin([T, C]) == C and vmin([U, C]) == U and vmin([F, T]) == F and vmin([]) == T
    assert vmax([F, U]) == U and vmax([U, C]) == C and vmax([C, T]) == T and vmax([]) == F
    assert [negate(v) for v in (T, F, U, C)] == [F, T, U, C]
    assert [negate_as_failure(v) for v in (T, F, U, C)] == [F, T, T, C]


# ── evidence alone (evaluateGroundAtom step 1 and 3) ──────────────────────────

@pytest.mark.parametrize("world,signs,expected", [
    ("open", "+", T), ("open", "-", F), ("open", "+-", C), ("open", "", U),
    ("closed", "+", T), ("closed", "-", F), ("closed", "+-", C), ("closed", "", F),
])
def test_evidence_alone(world, signs, expected, people):
    p = (predicate if world == "open" else closed_predicate)("p", {"who": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(people["alice"])
    for s in signs:
        (kb.asserts if s == "+" else kb.denies)(p(who=people["alice"]), source=f"src{s}")
    j = query(kb, p(who=people["alice"]))
    assert j.value == expected
    if expected == C:
        assert {l.sign for l in j.literals} == {"+", "-"} and j.reason == "conflicting evidence"
    if world == "closed" and not signs:
        assert j.reason == "closed-world absence"


# ── closed-world negation: the complete table (closed-world-negation case) ───

def test_closed_world_not_complete_table(people):
    sanction = closed_predicate("active_sanction", {"person": entity("person")})
    eligible = predicate("eligible", {"person": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(*people.values())
    kb.asserts(sanction(person=people["bob"]), source="sanctions_registry")
    kb.denies(sanction(person=people["cara"]), source="sanctions_registry")
    kb.asserts(sanction(person=people["drew"]), source="sanctions_registry")
    kb.denies(sanction(person=people["drew"]), source="appeals_board")
    X = variable("P")
    kb.rule(eligible(person=X)).iff(not_(sanction(person=X)))
    got = {n: query(kb, eligible(person=e)).value for n, e in people.items()}
    # missing -> true, asserted -> false, denied -> true, conflicted -> conflicted
    assert got == {"alice": T, "bob": F, "cara": T, "drew": C}


def test_open_world_not_keeps_unknown(people):
    seen = predicate("seen", {"person": entity("person")})
    unseen = predicate("unseen", {"person": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(people["alice"], people["bob"])
    kb.denies(seen(person=people["bob"]), source="log")
    X = variable("P")
    kb.rule(unseen(person=X)).iff(not_(seen(person=X)))
    assert query(kb, unseen(person=people["alice"])).value == U     # explicit negation of unknown
    assert query(kb, unseen(person=people["bob"])).value == T


# ── if vs iff, multiple rules, evidence/rule conflict (judgments.md) ─────────

@pytest.mark.parametrize("kind,body_value,expected", [
    ("if", T, T), ("if", C, C), ("if", U, U), ("if", F, U),
    ("iff", T, T), ("iff", C, C), ("iff", U, U), ("iff", F, F),
])
def test_rule_contribution_table(kind, body_value, expected):
    a = predicate("a")
    h = predicate("h")
    kb = KnowledgeBase()
    if body_value in (T, C):
        kb.asserts(a(), source="s1")
    if body_value in (F, C):
        kb.denies(a(), source="s2")
    getattr(kb.rule(h()), "if_" if kind == "if" else "iff")(a())
    j = query(kb, h())
    assert j.value == expected
    if expected == U and body_value == F:
        assert j.reason == "inconclusive rules"


def test_multiple_if_rules_take_the_strongest_route():
    a, b, h = predicate("a"), predicate("b"), predicate("h")
    kb = KnowledgeBase()
    kb.denies(a(), source="s")          # route 1 body false -> unknown
    kb.asserts(b(), source="s")         # route 2 body true
    kb.rule(h()).if_(a())
    kb.rule(h()).if_(b())
    j = query(kb, h())
    assert j.value == T and len(j.given) == 1 and repr(j.given[0].given[0].goal) == "b()"


def test_evidence_against_a_rule_is_conflicted():
    a, h = predicate("a"), predicate("h")
    kb = KnowledgeBase()
    kb.denies(a(), source="s")
    kb.rule(h()).iff(a())                # rule says false
    kb.asserts(h(), source="override")   # evidence says true
    j = query(kb, h())
    assert j.value == C and j.reason == "conflict between evidence and rules"
    assert [l.source for l in j.literals] == ["override"] and j.given[0].value == F


def test_evidence_and_agreeing_rule_is_true_with_both_in_the_proof():
    a, h = predicate("a"), predicate("h")
    kb = KnowledgeBase()
    kb.asserts(a(), source="s")
    kb.rule(h()).if_(a())
    kb.asserts(h(), source="direct")
    j = query(kb, h())
    assert j.value == T and j.reason == "evidence and rules"
    assert [l.source for l in j.literals] == ["direct"] and j.given[0].rule is not None


# ── and / or (operators.md; unknown-conjunction-subgoals, or-environments) ──

@pytest.mark.parametrize("left,right,and_v,or_v", [
    (T, T, T, T), (T, U, U, T), (T, F, F, T), (T, C, C, T),
    (U, F, F, U), (U, C, U, C), (F, C, F, C), (U, U, U, U),
])
def test_and_is_min_or_is_max(left, right, and_v, or_v):
    a, b, both, either = predicate("a"), predicate("b"), predicate("both"), predicate("either")
    kb = KnowledgeBase()
    for p, v in ((a, left), (b, right)):
        if v in (T, C):
            kb.asserts(p(), source="s")
        if v in (F, C):
            kb.denies(p(), source="s")
    kb.rule(both()).iff(and_(a(), b()))
    kb.rule(either()).iff(or_(a(), b()))
    assert (query(kb, both()).value, query(kb, either()).value) == (and_v, or_v)


# ── variables: head binding, existential body variables, joins ──────────────

def test_body_variable_is_existential_and_joins_across_atoms(people):
    manages = predicate("manages", {"boss": entity("person"), "who": entity("person")})
    approved = predicate("approved", {"who": entity("person")})
    can_sign = predicate("can_sign", {"boss": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(*people.values())
    kb.asserts(manages(boss=people["alice"], who=people["bob"]), source="org")
    kb.asserts(manages(boss=people["alice"], who=people["cara"]), source="org")
    kb.asserts(approved(who=people["cara"]), source="hr")
    kb.asserts(approved(who=people["drew"]), source="hr")       # drew is not managed by alice
    B, W = variable("B"), variable("W")
    kb.rule(can_sign(boss=B)).if_(and_(manages(boss=B, who=W), approved(who=W)))
    j = query(kb, can_sign(boss=people["alice"]))
    assert j.value == T
    assert j.given[0].given[0].reason.startswith("witness W=person:cara")
    assert query(kb, can_sign(boss=people["bob"])).value == U       # no witness: if-rule -> unknown


def test_enumeration_parameter_scenario_from_the_readme():
    """The accredited-investor walkthrough: three income claims, two of them
    over a year enumeration, combined by one rule."""
    alice = Entity("person", "alice")
    income = predicate("has_income_over_200k", {"person": entity("person"),
                                                "year": enumeration(2024, 2025)})
    expects = predicate("expects_same_income", {"person": entity("person")})
    qualified = predicate("is_qualified_investor", {"person": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(alice)
    P = variable("person")
    kb.rule(qualified(person=P)).if_(and_(income(person=P, year=2024),
                                          income(person=P, year=2025), expects(person=P)))
    kb.asserts(income(person=alice, year=2024), source="alice-2024-return#0-1", probability=0.9)
    kb.asserts(income(person=alice, year=2025), source="alice-2025-return#0-1", probability=0.8)
    assert query(kb, qualified(person=alice)).value == U            # expectation missing
    lit = kb.asserts(expects(person=alice), source="employer-letter#0-1", probability=0.7)
    assert query(kb, qualified(person=alice)).value == T
    kb.denies(income(person=alice, year=2025), source="alice-2025-original#0-1", probability=0.6)
    assert query(kb, qualified(person=alice)).value == C
    assert kb.retract(lit) is True and kb.retract(lit) is False
    assert query(kb, qualified(person=alice)).value == U
    with pytest.raises(ProgrammerError):
        income(person=alice, year=2023)


# ── recursion stops at the loop (mutual-recursion-cycle, unconstructive) ─────

def test_recursion_ends_at_the_loop_check():
    a, b = predicate("a"), predicate("b")
    ca, cb = closed_predicate("ca"), closed_predicate("cb")
    kb = KnowledgeBase()
    kb.rule(a()).if_(b())
    kb.rule(b()).if_(a())
    kb.rule(ca()).if_(cb())
    kb.rule(cb()).if_(ca())
    # a pure cycle of if-rules stays unknown in either world: the loop cut gives
    # closed-world absence (false) to the inner atom, and `if` turns a false
    # body into an inconclusive contribution (Factbag docs/formalized/recursion.md)
    assert query(kb, a()).value == U and query(kb, ca()).value == U
    kb.asserts(a(), source="s")
    assert query(kb, a()).value == T and query(kb, b()).value == T
    # an iff cycle of closed-world predicates settles at false
    da, db = closed_predicate("da"), closed_predicate("db")
    kb.rule(da()).iff(db())
    kb.rule(db()).iff(da())
    assert query(kb, da()).value == F and query(kb, da()).reason == "rules"


# ── DSL errors (unsafe-unbound-variable, exclusive-definition, schema) ───────

def test_dsl_errors(people):
    p = predicate("p", {"who": entity("person"), "n": int})
    h = predicate("h", {"who": entity("person")})
    kb = KnowledgeBase()
    X, N = variable("X"), variable("N")
    with pytest.raises(ProgrammerError, match="unsafe"):
        kb.rule(h(who=X)).if_(p(who=X, n=N))                 # N has no finite domain
    kb.rule(h(who=X)).if_(p(who=X, n=1))
    with pytest.raises(ProgrammerError, match="iff rule must be the only"):
        kb.rule(h(who=X)).iff(p(who=X, n=2))
    with pytest.raises(ProgrammerError, match="fields are"):
        p(who=people["alice"])
    with pytest.raises(ProgrammerError, match="is not a"):
        p(who="alice", n=1)
    with pytest.raises(ProgrammerError, match="ground"):
        kb.asserts(p(who=X, n=1), source="s")
    with pytest.raises(ProgrammerError, match="ground atom"):
        query(kb, h(who=X))
    with pytest.raises(ProgrammerError):
        enumeration(1, 1)
    with pytest.raises(ProgrammerError):
        enumeration(1, "a")


# ── fork, persistence, explanation ──────────────────────────────────────────

def test_fork_is_isolated(people):
    p = predicate("p", {"who": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(people["alice"])
    fork = kb.fork()
    fork.asserts(p(who=people["alice"]), source="later")
    assert query(kb, p(who=people["alice"])).value == U
    assert query(fork, p(who=people["alice"])).value == T
    fork.retract(fork.literals()[0])
    kb.asserts(p(who=people["alice"]), source="now")
    assert len(kb.literals()) == 1 and fork.literals() == []


def test_to_dict_round_trip_preserves_judgments(people):
    income = predicate("income", {"person": entity("person"), "year": enumeration(2024, 2025)},
                       description="Income over 200k in the year.")
    reviewed = closed_predicate("reviewed", {"person": entity("person"), "on": dt.date})
    ready = predicate("ready", {"person": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(Entity("person", "alice", name="Alice"))
    P = variable("P")
    kb.rule(ready(person=P)).if_(and_(income(person=P, year=2024), not_(reviewed(person=P, on=dt.date(2026, 1, 1)))))
    kb.asserts(income(person=people["alice"], year=2024), source="doc#0-9", probability=0.91,
               jev={"choice": "ASSERT"})
    kb.denies(income(person=people["alice"], year=2025), source="doc#9-20")
    data = kb.to_dict()
    kb2 = KnowledgeBase.from_dict(data)
    assert kb2.to_dict() == data
    inc2, ready2 = kb2.predicate("income"), kb2.predicate("ready")
    assert inc2.description == "Income over 200k in the year." and kb2.predicate("reviewed").world == "closed"
    a2 = kb2.entities.get("person", "alice")
    assert a2.attrs == {"name": "Alice"}
    assert query(kb2, ready2(person=a2)).value == query(kb, ready(person=people["alice"])).value == T
    assert query(kb2, inc2(person=a2, year=2025)).value == F
    assert [l.id for l in kb2.literals()] == [l.id for l in kb.literals()]
    assert kb2.literals()[0].metadata == {"jev": {"choice": "ASSERT"}}
    # a caller's own predicate object wins over the stored declaration
    kb3 = KnowledgeBase.from_dict(data, predicates=[income, reviewed, ready])
    assert kb3.predicate("income") is income


def test_formula_dict_shorthand():
    a, b, c = predicate("a"), predicate("b"), predicate("c")
    preds = {"a": a, "b": b, "c": c}
    f = formula_from_dict({"and": ["a", {"not": "b"}, {"or": ["c", {"atom": {"predicate": "a"}}]}]}, preds)
    assert isinstance(f, And) and formula_to_dict(f) == {
        "and": [{"atom": {"predicate": "a", "fields": {}}},
                {"not": {"atom": {"predicate": "b", "fields": {}}}},
                {"or": [{"atom": {"predicate": "c", "fields": {}}},
                        {"atom": {"predicate": "a", "fields": {}}}]}]}
    with pytest.raises(ProgrammerError, match="unknown claim"):
        formula_from_dict("zzz", preds)
    with pytest.raises(ProgrammerError, match="operator"):
        formula_from_dict({"xor": ["a", "b"]}, preds)


def test_explain_tree_carries_sources_and_rule_bindings(people):
    income = predicate("income", {"person": entity("person")})
    ready = predicate("ready", {"person": entity("person")})
    kb = KnowledgeBase()
    kb.entities.add(people["alice"])
    P = variable("person")
    kb.rule(ready(person=P)).if_(income(person=P))
    kb.asserts(income(person=people["alice"]), source="return#0-1", probability=0.93)
    tree = explain(query(kb, ready(person=people["alice"])))
    assert tree["claim"] == "ready(person:alice) = true" and tree["because"] == "rules"
    rule_node = tree["given"][0]
    assert rule_node["because"] == "ready(?person) if income(?person), with person = person:alice"
    leaf = rule_node["given"][0]
    assert leaf["claim"] == "income(person:alice) = true" and leaf["because"] == "evidence"
    assert leaf["evidence"] == [{"claim": "income(person:alice) = true", "sign": "+",
                                 "source": "return#0-1", "probability": 0.93, "id": "L1"}]


# ── a small reference check of the ground-atom algorithm ────────────────────

def _reference(ev_signs, rules):
    """evaluateGroundAtom by hand for a field-less head: ev_signs in {"", "+",
    "-", "+-"}; rules = [(kind, body_value)]."""
    values, positive, negative, inconclusive = [], 0, 0, 0
    if ev_signs == "+-":
        values.append(C)
    elif ev_signs == "+":
        values.append(T)
    elif ev_signs == "-":
        values.append(F)
    for kind, body in rules:
        c = body if kind == "iff" else (U if body == F else body)
        if c == U:
            inconclusive += 1
            continue
        values.append(c)
        positive += c in (T, C)
        negative += c in (F, C)
    if not values:
        return U if inconclusive else U
    if ("+" in ev_signs and negative) or ("-" in ev_signs and positive):
        return C
    return vmax(values)


def test_random_programs_match_the_reference():
    rng = random.Random(7)
    for _ in range(150):
        n_rules = rng.randint(0, 3)
        ev = rng.choice(["", "+", "-", "+-"])
        kinds = ["if"] * n_rules
        if n_rules == 1 and rng.random() < 0.5:
            kinds = ["iff"]
        bodies = [rng.choice([T, F, U, C]) for _ in range(n_rules)]
        h = predicate("h")
        kb = KnowledgeBase()
        for s in ev:
            (kb.asserts if s == "+" else kb.denies)(h(), source="e")
        for i, (kind, body) in enumerate(zip(kinds, bodies)):
            b = predicate(f"b{i}")
            if body in (T, C):
                kb.asserts(b(), source="s")
            if body in (F, C):
                kb.denies(b(), source="s")
            getattr(kb.rule(h()), "if_" if kind == "if" else "iff")(b())
        assert query(kb, h()).value == _reference(ev, list(zip(kinds, bodies))), (ev, kinds, bodies)
