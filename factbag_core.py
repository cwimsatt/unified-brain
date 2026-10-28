#!/usr/bin/env python3

# unified-brain — cheaper, faster more accurate AI coding agents, by DevLoops
# Copyright (C) 2026 DevLoops SINGLE MEMBER P.C.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# The semantics in this module are a Python port of Factbag
# (https://github.com/factbag/factbag, Apache-2.0, Copyright Factbag
# contributors): the four-valued lattice, the evidence/rule combination,
# open- and closed-world absence and the proof tree. See NOTICE.

"""
factbag_core — facts, rules and four-valued conclusions with provenance
-----------------------------------------------------------------------
A knowledge base holds *literals* (signed ground atoms, each with the source
it came from) and *rules*. A query answers one ground atom with one of four
values and the proof behind it:

    false < unknown < conflicted < true

How a ground atom gets its value (Factbag `evaluateGroundAtom`):

  1. evidence: positive and negative literals -> conflicted; only positive ->
     true; only negative -> false.
  2. every rule whose head matches contributes its body value: an `if` rule
     turns a false body into unknown (a failed sufficient condition says
     nothing), an `iff` rule passes the value through. An unknown contribution
     is inconclusive and adds no value. Rules are skipped for an atom that is
     already being evaluated higher up (recursion stops at the loop).
  3. no value at all: inconclusive rules -> unknown; a closed-world predicate
     -> false (closed-world absence); else unknown.
  4. evidence and rules that disagree -> conflicted; otherwise the maximum.

Rule bodies: `and` is the minimum, `or` the maximum. `not` of a closed-world
atom is negation-as-failure (unknown -> true); any other `not` swaps true and
false and leaves unknown and conflicted alone. Variables are bound by the
head; a body variable the head does not bind is existential: the body is
evaluated for every value of the variable's finite domain (an enumeration,
the booleans, or the registered entities of a type) and the best value wins.

Not ported from Factbag: arithmetic and comparison terms, aggregates,
let-bindings, pattern queries (queryAll) and the tabled fixed point for
recursive closed-world rules (recursion here ends at the loop check).
"""
from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ── Values ──────────────────────────────────────────────────────────────────

VALUES = ("false", "unknown", "conflicted", "true")
_RANK = {v: i for i, v in enumerate(VALUES)}


def vmin(values: Iterable[str]) -> str:
    """Conjunction. Empty -> true."""
    rank = 3
    for v in values:
        rank = min(rank, _RANK[v])
    return VALUES[rank]


def vmax(values: Iterable[str]) -> str:
    """Disjunction. Empty -> false."""
    rank = 0
    for v in values:
        rank = max(rank, _RANK[v])
    return VALUES[rank]


def negate(value: str) -> str:
    """Explicit negation: swaps true and false."""
    return {"true": "false", "false": "true"}.get(value, value)


def negate_as_failure(value: str) -> str:
    """Closed-world negation: unknown counts as absent, so it becomes true."""
    return {"true": "false", "false": "true", "unknown": "true"}.get(value, value)


# ── Schema: field types, entities, predicates, atoms ────────────────────────

class ProgrammerError(Exception):
    """A misuse of the DSL: wrong field, unsafe variable, invalid rule."""


class Enumeration:
    """A finite domain of plain values (str, int, float, bool, date)."""

    def __init__(self, members):
        members = tuple(members)
        if not members:
            raise ProgrammerError("an enumeration needs at least one member")
        kinds = {type(m) for m in members}
        if len(kinds) != 1:
            raise ProgrammerError(f"an enumeration has one member type, got {sorted(k.__name__ for k in kinds)}")
        if len(set(members)) != len(members):
            raise ProgrammerError("an enumeration has no duplicate members")
        self.members = members

    def __contains__(self, value) -> bool:
        return any(value == m and type(value) is type(m) for m in self.members)

    def __repr__(self) -> str:
        return f"enumeration{self.members!r}"


def enumeration(*members) -> Enumeration:
    return Enumeration(members)


class EntityType:
    """The type of an entity-valued field; its domain is kb.entities.of(name)."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"entity({self.name!r})"


def entity(name: str) -> EntityType:
    return EntityType(name)


class Entity:
    """Something the knowledge base talks about: a type name, an id and
    optional attributes (the attributes are shown to the classifier)."""

    __slots__ = ("type", "id", "attrs")

    def __init__(self, type: str, id: str, **attrs):
        if not type or not id:
            raise ProgrammerError("an entity needs a type and an id")
        self.type, self.id, self.attrs = str(type), str(id), dict(attrs)

    def __eq__(self, other) -> bool:
        return isinstance(other, Entity) and (self.type, self.id) == (other.type, other.id)

    def __hash__(self) -> int:
        return hash((self.type, self.id))

    def __repr__(self) -> str:
        return f"{self.type}:{self.id}"


class Var:
    """A rule variable. Equal to itself only."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"?{self.name}"


def variable(name: str) -> Var:
    return Var(name)


_PLAIN = {bool: "bool", str: "str", int: "number", float: "number", _dt.date: "date"}


def _field_kind(spec) -> str:
    if spec is bool:
        return "bool"
    if spec is str:
        return "str"
    if spec in (int, float):
        return "number"
    if spec is _dt.date:
        return "date"
    if isinstance(spec, Enumeration):
        return "enum"
    if isinstance(spec, EntityType):
        return "entity"
    raise ProgrammerError(f"unsupported field type {spec!r}: use bool, str, int, float, "
                          "datetime.date, enumeration(...) or entity(...)")


def _check_value(pred: "Predicate", name: str, value) -> None:
    spec = pred.params[name]
    kind = pred.kinds[name]
    ok = {
        "bool": lambda v: type(v) is bool,
        "str": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and type(v) is not bool,
        "date": lambda v: isinstance(v, _dt.date),
        "enum": lambda v: v in spec,
        "entity": lambda v: isinstance(v, Entity) and v.type == spec.name,
    }[kind]
    if not ok(value):
        raise ProgrammerError(f"{pred.name}.{name}: {value!r} is not a {spec!r}")


def _value_key(value):
    """A hashable, type-tagged key for a field value (True is not 1 here)."""
    if isinstance(value, Entity):
        return ("e", value.type, value.id)
    if type(value) is bool:
        return ("b", value)
    if isinstance(value, (int, float)):
        return ("n", float(value))
    if isinstance(value, _dt.datetime):
        return ("d", value.isoformat())
    if isinstance(value, _dt.date):
        return ("d", value.isoformat())
    return ("s", str(value))


class Predicate:
    """A named relation over typed fields. Calling it builds an atom."""

    def __init__(self, name: str, params: Optional[dict] = None, *, world: str = "open",
                 description: str = "", metadata: Optional[dict] = None):
        if not name or not isinstance(name, str):
            raise ProgrammerError("a predicate needs a name")
        if world not in ("open", "closed"):
            raise ProgrammerError("world is 'open' or 'closed'")
        self.name = name
        self.params = dict(params or {})
        self.kinds = {k: _field_kind(v) for k, v in self.params.items()}
        self.world = world
        self.description = description
        self.metadata = dict(metadata or {})

    def __call__(self, **fields) -> "Atom":
        unknown = set(fields) - set(self.params)
        missing = set(self.params) - set(fields)
        if unknown or missing:
            raise ProgrammerError(f"{self.name}: fields are {sorted(self.params)}, "
                                  f"got {sorted(fields)}")
        for k, v in fields.items():
            if not isinstance(v, Var):
                _check_value(self, k, v)
        return Atom(self, {k: fields[k] for k in self.params})

    def __repr__(self) -> str:
        return f"<predicate {self.name}>"


def predicate(name: str, params: Optional[dict] = None, *, description: str = "",
              metadata: Optional[dict] = None) -> Predicate:
    """An open-world predicate: missing evidence leaves an atom unknown."""
    return Predicate(name, params, world="open", description=description, metadata=metadata)


def closed_predicate(name: str, params: Optional[dict] = None, *, description: str = "",
                     metadata: Optional[dict] = None) -> Predicate:
    """A closed-world predicate: missing evidence makes an atom false and
    `not` on it is negation-as-failure."""
    return Predicate(name, params, world="closed", description=description, metadata=metadata)


class Atom:
    __slots__ = ("predicate", "fields")

    def __init__(self, predicate: Predicate, fields: dict):
        self.predicate, self.fields = predicate, fields

    def is_ground(self) -> bool:
        return not any(isinstance(v, Var) for v in self.fields.values())

    def variables(self) -> list:
        return [v for v in self.fields.values() if isinstance(v, Var)]

    def key(self) -> tuple:
        if not self.is_ground():
            raise ProgrammerError(f"{self} is not ground")
        return (self.predicate.name,) + tuple((k, _value_key(v)) for k, v in self.fields.items())

    def substitute(self, bindings: dict) -> "Atom":
        out = {}
        for k, v in self.fields.items():
            if isinstance(v, Var) and v in bindings:
                v = bindings[v]
                _check_value(self.predicate, k, v)
            out[k] = v
        return Atom(self.predicate, out)

    def __eq__(self, other) -> bool:
        return (isinstance(other, Atom) and other.predicate is self.predicate
                and other.is_ground() and self.is_ground() and other.key() == self.key())

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        args = ", ".join(_show(v) for v in self.fields.values())
        return f"{self.predicate.name}({args})"


def _show(v) -> str:
    if isinstance(v, (Var, Entity)):
        return repr(v)
    if isinstance(v, _dt.date):
        return v.isoformat()
    return repr(v)


# ── Formulas ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class And:
    args: tuple


@dataclass(frozen=True)
class Or:
    args: tuple


@dataclass(frozen=True)
class Not:
    arg: Any


def _formula(x):
    if isinstance(x, (Atom, And, Or, Not)):
        return x
    raise ProgrammerError(f"not a formula: {x!r}")


def and_(*args):
    return And(tuple(_formula(a) for a in args))


def or_(*args):
    return Or(tuple(_formula(a) for a in args))


def not_(arg):
    return Not(_formula(arg))


def _formula_vars(f) -> list:
    """Variables of a formula, in first-seen order, without duplicates."""
    out: list = []
    def walk(x):
        if isinstance(x, Atom):
            for v in x.variables():
                if v not in out:
                    out.append(v)
        elif isinstance(x, (And, Or)):
            for a in x.args:
                walk(a)
        elif isinstance(x, Not):
            walk(x.arg)
    walk(f)
    return out


def _formula_var_specs(f, var: Var) -> list:
    """The (predicate, field) positions a variable appears in."""
    out = []
    def walk(x):
        if isinstance(x, Atom):
            for k, v in x.fields.items():
                if v is var:
                    out.append((x.predicate, k))
        elif isinstance(x, (And, Or)):
            for a in x.args:
                walk(a)
        elif isinstance(x, Not):
            walk(x.arg)
    walk(f)
    return out


def formula_str(f) -> str:
    if isinstance(f, Atom):
        return repr(f)
    if isinstance(f, And):
        return "and(" + ", ".join(formula_str(a) for a in f.args) + ")"
    if isinstance(f, Or):
        return "or(" + ", ".join(formula_str(a) for a in f.args) + ")"
    if isinstance(f, Not):
        return "not(" + formula_str(f.arg) + ")"
    return repr(f)


# ── Knowledge base ──────────────────────────────────────────────────────────

@dataclass
class Literal:
    id: str
    sign: str                      # "+" or "-"
    atom: Atom
    source: Optional[str] = None
    probability: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.sign}{self.atom!r}@{self.source}"


@dataclass
class Rule:
    head: Atom
    kind: str                      # "if" or "iff"
    body: Any

    def __repr__(self) -> str:
        return f"{self.head!r} {self.kind} {formula_str(self.body)}"


class EntityRegistry:
    def __init__(self):
        self._by_type: dict = {}

    def add(self, *entities: Entity) -> None:
        for e in entities:
            if not isinstance(e, Entity):
                raise ProgrammerError(f"not an Entity: {e!r}")
            self._by_type.setdefault(e.type, {})[e.id] = e

    def get(self, type: str, id: str) -> Optional[Entity]:
        return self._by_type.get(type, {}).get(id)

    def of(self, type: str) -> list:
        return list(self._by_type.get(type, {}).values())

    def all(self) -> list:
        return [e for d in self._by_type.values() for e in d.values()]

    def copy(self) -> "EntityRegistry":
        r = EntityRegistry()
        r._by_type = {t: dict(d) for t, d in self._by_type.items()}
        return r


class _RuleBuilder:
    def __init__(self, kb: "KnowledgeBase", head: Atom):
        self.kb, self.head = kb, head

    def if_(self, body) -> Rule:
        return self.kb._add_rule(Rule(self.head, "if", _formula(body)))

    def iff(self, body) -> Rule:
        return self.kb._add_rule(Rule(self.head, "iff", _formula(body)))


class KnowledgeBase:
    def __init__(self):
        self.entities = EntityRegistry()
        self._literals: list = []
        self._rules: list = []
        self._predicates: dict = {}
        self._seq = 0

    # -- writes --------------------------------------------------------------
    def _register(self, pred: Predicate) -> None:
        known = self._predicates.get(pred.name)
        if known is None:
            self._predicates[pred.name] = pred
        elif known is not pred:
            raise ProgrammerError(f"two different predicates named {pred.name!r}")

    def declare(self, *predicates: Predicate) -> None:
        """Register predicates up front (a predicate is also registered the
        first time evidence or a rule mentions it)."""
        for p in predicates:
            if not isinstance(p, Predicate):
                raise ProgrammerError(f"not a predicate: {p!r}")
            self._register(p)

    def _literal(self, sign: str, atom: Atom, source, probability, metadata) -> Literal:
        if not atom.is_ground():
            raise ProgrammerError(f"evidence must be ground: {atom}")
        self._register(atom.predicate)
        for v in atom.fields.values():
            if isinstance(v, Entity) and self.entities.get(v.type, v.id) is None:
                self.entities.add(v)
        self._seq += 1
        lit = Literal(f"L{self._seq}", sign, atom, source, probability, dict(metadata or {}))
        self._literals.append(lit)
        return lit

    def asserts(self, atom: Atom, *, source: Optional[str] = None,
                probability: Optional[float] = None, **metadata) -> Literal:
        return self._literal("+", atom, source, probability, metadata)

    def denies(self, atom: Atom, *, source: Optional[str] = None,
               probability: Optional[float] = None, **metadata) -> Literal:
        return self._literal("-", atom, source, probability, metadata)

    def rule(self, head: Atom) -> _RuleBuilder:
        return _RuleBuilder(self, head)

    def _add_rule(self, rule: Rule) -> Rule:
        head = rule.head
        self._register(head.predicate)
        for a in _atoms_of(rule.body):
            self._register(a.predicate)
        existing = [r for r in self._rules if r.head.predicate is head.predicate]
        if rule.kind == "iff" and existing or any(r.kind == "iff" for r in existing):
            raise ProgrammerError(f"{head.predicate.name}: an iff rule must be the only rule "
                                  "for its predicate")
        # Safety: every body variable must have a finite domain, and a
        # variable means one thing: every position it fills has the same type.
        head_vars = set(head.variables())
        for v in _formula_vars(rule.body) + list(head_vars):
            specs = _formula_var_specs(rule.body, v) + [(head.predicate, k) for k, x in head.fields.items() if x is v]
            kinds = {p.kinds[k] for p, k in specs}
            if v not in head_vars and kinds & {"str", "number", "date"}:
                raise ProgrammerError(f"{v!r} in {rule!r} is unsafe: it is not bound by the "
                                      "head and has no finite domain")
            if len(kinds) > 1 or len({_spec_key(p.params[k]) for p, k in specs}) > 1:
                raise ProgrammerError(f"{v!r} in {rule!r} is used with different types")
        self._rules.append(rule)
        return rule

    def retract(self, target) -> bool:
        """Remove one literal by id or Literal. False when it is not there."""
        lid = target.id if isinstance(target, Literal) else str(target)
        before = len(self._literals)
        self._literals = [l for l in self._literals if l.id != lid]
        return len(self._literals) < before

    # -- reads ---------------------------------------------------------------
    def literals(self, predicate: Optional[Predicate] = None, *, sign: Optional[str] = None,
                 source: Optional[str] = None) -> list:
        out = self._literals
        if predicate is not None:
            out = [l for l in out if l.atom.predicate is predicate]
        if sign is not None:
            out = [l for l in out if l.sign == sign]
        if source is not None:
            out = [l for l in out if l.source == source]
        return list(out)

    def rules(self) -> list:
        return list(self._rules)

    def predicates(self) -> list:
        return list(self._predicates.values())

    def predicate(self, name: str) -> Optional[Predicate]:
        return self._predicates.get(name)

    def fork(self) -> "KnowledgeBase":
        """A copy with its own literal and rule lists; literals, rules,
        predicates and entities are shared objects."""
        kb = KnowledgeBase()
        kb.entities = self.entities.copy()
        kb._literals = list(self._literals)
        kb._rules = list(self._rules)
        kb._predicates = dict(self._predicates)
        kb._seq = self._seq
        return kb

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "predicates": [_pred_to_dict(p) for p in sorted(self._predicates.values(), key=lambda p: p.name)],
            "entities": [{"type": e.type, "id": e.id, "attrs": e.attrs} for e in self.entities.all()],
            "literals": [{"id": l.id, "sign": l.sign, "predicate": l.atom.predicate.name,
                          "fields": _fields_to_dict(l.atom.fields), "source": l.source,
                          "probability": l.probability, "metadata": l.metadata}
                         for l in self._literals],
            "rules": [{"head": {"predicate": r.head.predicate.name,
                                "fields": _fields_to_dict(r.head.fields)},
                       "kind": r.kind, "body": formula_to_dict(r.body)} for r in self._rules],
        }

    @classmethod
    def from_dict(cls, data: dict, predicates: Optional[Iterable[Predicate]] = None) -> "KnowledgeBase":
        """Rebuild a knowledge base. Predicates are re-created from their stored
        declarations unless the caller passes its own (matched by name)."""
        kb = cls()
        preds = {p.name: p for p in (predicates or [])}
        for d in data.get("predicates", []):
            preds.setdefault(d["name"], _pred_from_dict(d))
        for e in data.get("entities", []):
            kb.entities.add(Entity(e["type"], e["id"], **(e.get("attrs") or {})))
        vars_: dict = {}
        for d in data.get("literals", []):
            pred = preds[d["predicate"]]
            atom = pred(**_fields_from_dict(pred, d["fields"], kb, vars_))
            lit = kb._literal(d["sign"], atom, d.get("source"), d.get("probability"),
                              d.get("metadata"))
            lit.id = d.get("id") or lit.id
        kb._seq = max([kb._seq] + [int(l.id[1:]) for l in kb._literals if l.id[1:].isdigit()])
        for d in data.get("rules", []):
            pred = preds[d["head"]["predicate"]]
            head = pred(**_fields_from_dict(pred, d["head"]["fields"], kb, vars_))
            body = formula_from_dict(d["body"], preds, kb, vars_)
            kb._add_rule(Rule(head, d["kind"], body))
        return kb


def _spec_key(spec):
    if isinstance(spec, Enumeration):
        return ("enum",) + tuple(_value_key(m) for m in spec.members)
    if isinstance(spec, EntityType):
        return ("entity", spec.name)
    return (_field_kind(spec),)


def _atoms_of(f) -> list:
    if isinstance(f, Atom):
        return [f]
    if isinstance(f, (And, Or)):
        return [a for x in f.args for a in _atoms_of(x)]
    if isinstance(f, Not):
        return _atoms_of(f.arg)
    return []


def _pred_to_dict(p: Predicate) -> dict:
    params = {}
    for k, spec in p.params.items():
        kind = p.kinds[k]
        d: dict = {"type": kind}
        if kind == "enum":
            d["members"] = [_value_to_json(m) for m in spec.members]
        elif kind == "entity":
            d["entity"] = spec.name
        params[k] = d
    return {"name": p.name, "world": p.world, "description": p.description,
            "metadata": p.metadata, "params": params}


def _pred_from_dict(d: dict) -> Predicate:
    params = {}
    for k, spec in (d.get("params") or {}).items():
        t = spec["type"]
        if t == "enum":
            params[k] = Enumeration(_value_from_json(m) for m in spec["members"])
        elif t == "entity":
            params[k] = EntityType(spec["entity"])
        else:
            params[k] = {"bool": bool, "str": str, "number": float, "date": _dt.date}[t]
    return Predicate(d["name"], params, world=d.get("world", "open"),
                     description=d.get("description", ""), metadata=d.get("metadata"))


def _value_to_json(v):
    if isinstance(v, Entity):
        return {"entity": [v.type, v.id]}
    if isinstance(v, Var):
        return {"var": v.name}
    if isinstance(v, _dt.datetime):
        return {"datetime": v.isoformat()}
    if isinstance(v, _dt.date):
        return {"date": v.isoformat()}
    return v


def _value_from_json(v, kb: Optional[KnowledgeBase] = None, vars_: Optional[dict] = None):
    if isinstance(v, dict):
        if "entity" in v:
            t, i = v["entity"]
            found = kb.entities.get(t, i) if kb else None
            return found or Entity(t, i)
        if "var" in v:
            vars_ = vars_ if vars_ is not None else {}
            return vars_.setdefault(v["var"], Var(v["var"]))
        if "date" in v:
            return _dt.date.fromisoformat(v["date"])
        if "datetime" in v:
            return _dt.datetime.fromisoformat(v["datetime"])
    return v


def _fields_to_dict(fields: dict) -> dict:
    return {k: _value_to_json(v) for k, v in fields.items()}


def _fields_from_dict(pred: Predicate, fields: dict, kb: KnowledgeBase, vars_: dict) -> dict:
    out = {}
    for k, v in fields.items():
        v = _value_from_json(v, kb, vars_)
        if pred.kinds.get(k) == "number" and isinstance(v, (int, float)) and type(v) is not bool:
            v = float(v) if isinstance(pred.params[k], type) else v
        out[k] = v
    return out


def atom_from_dict(d, predicates: dict, kb: Optional[KnowledgeBase] = None,
                   vars_: Optional[dict] = None) -> Atom:
    """An atom from a claim name or {"predicate": name, "fields": {...}};
    {"var": name} field values become variables shared through `vars_`."""
    vars_ = vars_ if vars_ is not None else {}
    kb = kb or KnowledgeBase()
    if isinstance(d, str):
        name, fields = d, {}
    elif isinstance(d, dict) and isinstance(d.get("predicate"), str):
        name, fields = d["predicate"], d.get("fields") or {}
    else:
        raise ProgrammerError(f"an atom is a claim name or {{\"predicate\", \"fields\"}}, got {d!r}")
    if name not in predicates:
        raise ProgrammerError(f"unknown claim {name!r}")
    pred = predicates[name]
    return pred(**_fields_from_dict(pred, fields, kb, vars_))


def formula_to_dict(f) -> dict:
    if isinstance(f, Atom):
        return {"atom": {"predicate": f.predicate.name, "fields": _fields_to_dict(f.fields)}}
    if isinstance(f, And):
        return {"and": [formula_to_dict(a) for a in f.args]}
    if isinstance(f, Or):
        return {"or": [formula_to_dict(a) for a in f.args]}
    if isinstance(f, Not):
        return {"not": formula_to_dict(f.arg)}
    raise ProgrammerError(f"not a formula: {f!r}")


def formula_from_dict(d, predicates: dict, kb: Optional[KnowledgeBase] = None,
                      vars_: Optional[dict] = None):
    """The inverse of formula_to_dict. Shorthands: a bare string is an atom of
    a field-less predicate; {"not": "x"}, {"and": [...]}, {"or": [...]}."""
    vars_ = vars_ if vars_ is not None else {}
    kb = kb or KnowledgeBase()
    if isinstance(d, str):
        if d not in predicates:
            raise ProgrammerError(f"unknown claim {d!r}")
        return predicates[d]()
    if not isinstance(d, dict) or len(d) != 1:
        raise ProgrammerError(f"a formula is a string or a one-key object, got {d!r}")
    (k, v), = d.items()
    if k == "and":
        return And(tuple(formula_from_dict(x, predicates, kb, vars_) for x in v))
    if k == "or":
        return Or(tuple(formula_from_dict(x, predicates, kb, vars_) for x in v))
    if k == "not":
        return Not(formula_from_dict(v, predicates, kb, vars_))
    if k == "atom":
        return atom_from_dict(v, predicates, kb, vars_)
    raise ProgrammerError(f"unknown formula operator {k!r}")


# ── Solver ──────────────────────────────────────────────────────────────────

@dataclass
class Judgment:
    """A value with its proof. `given` are the sub-judgments the value rests
    on; `literals` the evidence read at this node; `rule` the rule applied."""
    value: str
    goal: Any                              # Atom or formula
    reason: str = ""
    given: list = field(default_factory=list)
    literals: list = field(default_factory=list)
    rule: Optional[Rule] = None
    bindings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{formula_str(self.goal)} = {self.value}>"


class _Index:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.literals: dict = {}
        for l in kb._literals:
            self.literals.setdefault(l.atom.key(), ([], []))[0 if l.sign == "+" else 1].append(l)
        self.rules: dict = {}
        for r in kb._rules:
            self.rules.setdefault(r.head.predicate.name, []).append(r)

    def evidence(self, atom: Atom):
        return self.literals.get(atom.key(), ([], []))


MAX_WITNESSES = 10_000


def _domain(kb: KnowledgeBase, pred: Predicate, name: str) -> list:
    kind = pred.kinds[name]
    if kind == "bool":
        return [False, True]
    if kind == "enum":
        return list(pred.params[name].members)
    if kind == "entity":
        return kb.entities.of(pred.params[name].name)
    raise ProgrammerError(f"{pred.name}.{name} has no finite domain to enumerate")


def _unify(head: Atom, atom: Atom) -> Optional[dict]:
    if head.predicate is not atom.predicate:
        return None
    bindings: dict = {}
    for k, hv in head.fields.items():
        av = atom.fields[k]
        if isinstance(hv, Var):
            if hv in bindings:
                if _value_key(bindings[hv]) != _value_key(av):
                    return None
            else:
                bindings[hv] = av
        elif _value_key(hv) != _value_key(av):
            return None
    return bindings


def _solve_atom(ix: _Index, atom: Atom, active: frozenset) -> Judgment:
    kb = ix.kb
    key = atom.key()
    pos, neg = ix.evidence(atom)
    values: list = []
    positive: list = []
    negative: list = []
    inconclusive: list = []
    if pos and neg:
        values.append("conflicted")
    elif pos:
        values.append("true")
    elif neg:
        values.append("false")

    if key not in active:
        inner = active | {key}
        for rule in ix.rules.get(atom.predicate.name, []):
            bindings = _unify(rule.head, atom)
            if bindings is None:
                continue
            body = _solve_body(ix, rule.body, bindings, inner)
            contribution = body.value if rule.kind == "iff" else (
                "unknown" if body.value == "false" else body.value)
            support = Judgment(contribution, atom, f"rule {rule.kind}", [body], rule=rule,
                               bindings=bindings)
            if contribution == "unknown":
                inconclusive.append(support)
                continue
            values.append(contribution)
            if contribution in ("true", "conflicted"):
                positive.append(support)
            if contribution in ("false", "conflicted"):
                negative.append(support)

    if not values:
        if inconclusive:
            return Judgment("unknown", atom, "inconclusive rules", inconclusive)
        if atom.predicate.world == "closed":
            return Judgment("false", atom, "closed-world absence")
        return Judgment("unknown", atom, "no evidence" if key not in active else "recursion")

    conflict = (pos and negative) or (neg and positive)
    value = "conflicted" if conflict else vmax(values)
    given: list = []
    literals: list = []
    if value in ("true", "conflicted"):
        literals += pos
        given += positive
    if value in ("false", "conflicted"):
        literals += neg
        given += negative
    reason = "evidence" if literals and not given else ("rules" if given and not literals
                                                         else "evidence and rules")
    if conflict:
        reason = "conflict between evidence and rules"
    elif pos and neg and value == "conflicted":
        reason = "conflicting evidence"
    return Judgment(value, atom, reason, given, literals)


def _solve_body(ix: _Index, body, bindings: dict, active: frozenset) -> Judgment:
    """A rule body under the head's bindings. Variables the head leaves free
    are existential: every assignment over their domains is tried and the
    best one is kept."""
    free = [v for v in _formula_vars(body) if v not in bindings]
    if not free:
        return _solve_formula(ix, body, bindings, active)
    domains = []
    for v in free:
        pred, name = _formula_var_specs(body, v)[0]
        domains.append(_domain(ix.kb, pred, name))
    total = 1
    for d in domains:
        total *= len(d)
    if total > MAX_WITNESSES:
        raise ProgrammerError(f"{total} witness combinations for {[v.name for v in free]}; "
                              f"the limit is {MAX_WITNESSES}")
    best: Optional[Judgment] = None
    tried = 0
    for combo in itertools.product(*domains):
        tried += 1
        b = dict(bindings)
        b.update(zip(free, combo))
        j = _solve_formula(ix, body, b, active)
        if best is None or _RANK[j.value] > _RANK[best.value]:
            best = j
            if best.value == "true":
                break
    if best is None:                       # an empty domain
        return Judgment("false", body, "no witness", bindings=dict(bindings))
    best.reason = (f"witness {', '.join(f'{v.name}={_show(best.bindings[v])}' for v in free)}"
                   f" ({tried} tried)")
    return best


def _solve_formula(ix: _Index, f, bindings: dict, active: frozenset) -> Judgment:
    if isinstance(f, Atom):
        ground = f.substitute(bindings)
        j = _solve_atom(ix, ground, active)
        j.bindings = dict(bindings)
        return j
    if isinstance(f, And):
        parts = [_solve_formula(ix, a, bindings, active) for a in f.args]
        return Judgment(vmin(p.value for p in parts), f, "and: the weakest premise", parts,
                        bindings=dict(bindings))
    if isinstance(f, Or):
        parts = [_solve_formula(ix, a, bindings, active) for a in f.args]
        return Judgment(vmax(p.value for p in parts), f, "or: the strongest alternative", parts,
                        bindings=dict(bindings))
    if isinstance(f, Not):
        inner = _solve_formula(ix, f.arg, bindings, active)
        if isinstance(f.arg, Atom) and f.arg.predicate.world == "closed":
            return Judgment(negate_as_failure(inner.value), f, "negation as failure", [inner],
                            bindings=dict(bindings))
        return Judgment(negate(inner.value), f, "negation", [inner], bindings=dict(bindings))
    raise ProgrammerError(f"not a formula: {f!r}")


def query(kb: KnowledgeBase, atom: Atom) -> Judgment:
    """The value of one ground atom and the proof behind it."""
    if not isinstance(atom, Atom) or not atom.is_ground():
        raise ProgrammerError(f"query needs a ground atom, got {atom!r}")
    return _solve_atom(_Index(kb), atom, frozenset())


# ── Explanation ─────────────────────────────────────────────────────────────

def explain(j: Judgment) -> dict:
    """A JSON-friendly proof tree: {claim, value, because, given, evidence}.
    Evidence entries carry the literal's source and probability so a reader
    can go back to the passage."""
    goal = j.goal
    claim = f"{formula_str(goal)} = {j.value}"
    node: dict = {"claim": claim, "value": j.value}
    if j.rule is not None:
        node["because"] = (f"{j.rule!r}" + (", with " + ", ".join(
            f"{v.name} = {_show(x)}" for v, x in j.bindings.items()) if j.bindings else ""))
    elif j.reason:
        node["because"] = j.reason
    if j.literals:
        node["evidence"] = [{"claim": f"{l.atom!r} = {'true' if l.sign == '+' else 'false'}",
                             "sign": l.sign, "source": l.source, "probability": l.probability,
                             "id": l.id} for l in j.literals]
    if j.given:
        node["given"] = [explain(g) for g in j.given]
    return node
