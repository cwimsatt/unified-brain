"""
Shared fixtures. Everything runs against a synthetic three-commit repository
built in a temp dir; no network, no API key, Jev is a stub (jev_mock).

    tiny_repo     the repository (session)
    ingested_db   that repository ingested into a fresh database file (session)
    service       codebase_qa_service imported once against ingested_db (session)
    jev_mock      codebase_qa_service._jev_post replaced by an in-process stub
"""
import asyncio
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Fixture sources (also imported by test_extract.py) ───────────────────────

PY_V1 = '''import os


def hello():
    return "hello"


class Greeter:
    def greet(self):
        return hello()
'''

# the third commit: hello's body changes, its line does not
PY_V2 = PY_V1.replace('return "hello"', 'return "hello, world"')

README = "# tiny\n\nA repository the tests build.\n"

RS = '''struct Counter {
    n: u32,
}

impl Counter {
    pub fn inc(&mut self) {
        helper();
        println!("x");
    }
}

fn helper() {}

macro_rules! twice {
    ($e:expr) => { $e; $e };
}
'''

SCAD = '''SIZE = 10;

module box(w) {
    cube(w);
}

function area(w) = w * w;

function f(x) = area(x);
'''

BLOB = b"\x00\x01binary\x00\xff"
NON_ASCII_NAME = "données.txt"

# The environment the service is imported with. load_dotenv in the service
# is override=False, so these win over a .env in the checkout.
SERVICE_ENV = {
    "QA_VECTOR_STORE": "none",
    "JEV_PROVIDER": "typesafe",
    "TYPESAFE_API_KEY": "test",
    "LLM_PROVIDER": "llmbase",
    "LLM_API_KEY": "test",
    "JEV_EXPAND": "0",
    "JEV_HIT_TRIAGE": "0",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "CODE_EMBED_PASS2": "none",
    "UB_SKIP_GLOBS": "",
    # Pinned, not deleted: load_dotenv (override=False) only yields to
    # variables that are present, so a .env in the checkout could flip these.
    "UB_ALLOW_EMPTY_DB": "",          # the service compares it to "1"
    "GREP_MAX_RESULTS": "500",
    "JEV_MIN_SCORE": "2.0",
    # Present-but-empty: a .env in the checkout cannot fill them in, and the
    # service treats "" as unset (provider defaults, no code store loaded).
    "JEV_MODEL": "",
    "JEV_API_BASE": "",
    "JEV_API_KEY": "",
    "VOYAGE_API_KEY": "",
    "DEEPINFRA_API_KEY": "",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def git(repo, *args):
    """git in `repo` with a fixed identity; raises on failure."""
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", "-C", repo, *args],
        check=True, capture_output=True).stdout.decode("utf-8", "replace")


def _write(repo, rel, data):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as f:
        f.write(data)


@pytest.fixture(scope="session")
def tiny_repo(tmp_path_factory):
    """Three commits: python + readme; rust + openscad; an edit, a binary
    blob (NUL byte) and a non-ASCII file name."""
    repo = str(tmp_path_factory.mktemp("tiny_repo"))
    git(repo, "init", "-q")
    _write(repo, "pkg/app.py", PY_V1)
    _write(repo, "README.md", README)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "python app")
    _write(repo, "src/lib.rs", RS)
    _write(repo, "parts/box.scad", SCAD)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "rust and openscad")
    _write(repo, "pkg/app.py", PY_V2)
    _write(repo, "fixtures/blob.bin", BLOB)
    _write(repo, NON_ASCII_NAME, "café\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "edit hello, add blob and non-ascii name")
    return repo


@pytest.fixture(scope="session")
def session_env():
    """A MonkeyPatch that lives for the whole session (the service reads
    the Jev settings from the environment on every call)."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def ingested_db(tiny_repo, tmp_path_factory, session_env):
    """tiny_repo ingested into a fresh database file through next_claude's
    entry points (no embedding pass: CODE_EMBED_PASS2=none)."""
    session_env.setenv("CODE_EMBED_PASS2", "none")
    session_env.setenv("UB_SKIP_GLOBS", "")
    import next_claude
    db_path = str(tmp_path_factory.mktemp("db") / "tiny.unified-brain.db")
    r, rb = next_claude._db_clients(db_path)
    totals = asyncio.run(next_claude.ingest_repo(
        tiny_repo, since=None, max_n=None, r=r, rb=rb))
    assert totals["failed"] == 0, totals
    yield types.SimpleNamespace(path=db_path, r=r, rb=rb, totals=totals,
                                repo=tiny_repo)
    r.close()


@pytest.fixture(scope="session")
def service(ingested_db, session_env):
    """codebase_qa_service imported once, against ingested_db, with the
    key-less settings: no vector store, Jev at typesafe with a dummy key."""
    for k, v in SERVICE_ENV.items():
        session_env.setenv(k, v)
    session_env.setenv("DB_PATH", ingested_db.path)
    assert "codebase_qa_service" not in sys.modules, \
        "import codebase_qa_service only through the service fixture"
    import codebase_qa_service
    return codebase_qa_service


@pytest.fixture
def jev_mock(service, monkeypatch):
    """codebase_qa_service._jev_post replaced by a stub: the one seam every
    Jev call site goes through, threads included. The suite exercises
    `select` and, with JEV_EXPAND=1, `chain` (verdict + names in one call);
    triage (JEV_HIT_TRIAGE=0 here) and the probe pick use the same helper
    but are not driven. score questions answer 4.0 when the state's symbol
    is in `relevant`, else 0.0; choice answers 'reject' (so a chain stops at
    its seeds); noul answers 0.0. Claims questions (ASSERT/DENY/IRRELEVANT)
    answer from `claims`, resolver questions (KEEP/RETRACT) from `standing`.
    `fail` set to an error string makes every call fail with it."""
    mock = types.SimpleNamespace(relevant=set(), fail=None, calls=[],
                                 # claims: {claim name: {"assert": [substr...], "deny": [...]}}
                                 # -> ASSERT 0.9 / DENY 0.9 when a substring is in the
                                 # passage text, else IRRELEVANT 1.0
                                 claims={},
                                 # standing: substrings of passages the resolver RETRACTs
                                 standing=[])

    def _post(client, state, questions, *, what, attempts=4):
        mock.calls.append({"what": what, "state": dict(state),
                           "questions": questions})
        if mock.fail:
            return None, 0.0, mock.fail
        answers = {}
        for key, q in questions.items():
            kind = q.get("type")
            if kind == "score":
                answers[key] = {"score": 4.0 if state.get("symbol") in mock.relevant
                                else 0.0}
            elif kind == "choice" and set(q.get("criteria") or ()) == {"ASSERT", "DENY", "IRRELEVANT"}:
                name = q["instructions"]["predicate"]["name"]
                text = state["document"]["text"]
                spec = mock.claims.get(name, {})
                if any(s in text for s in spec.get("assert", [])):
                    p = {"ASSERT": 0.9, "DENY": 0.05, "IRRELEVANT": 0.05}
                elif any(s in text for s in spec.get("deny", [])):
                    p = {"ASSERT": 0.05, "DENY": 0.9, "IRRELEVANT": 0.05}
                else:
                    p = {"ASSERT": 0.0, "DENY": 0.0, "IRRELEVANT": 1.0}
                answers[key] = {"choice": max(p, key=p.get), "probabilities": p}
            elif kind == "choice" and set(q.get("criteria") or ()) == {"KEEP", "RETRACT"}:
                text = state["documents"][int(key.split("_")[1])]["text"]
                retract = any(s in text for s in mock.standing)
                answers[key] = {"choice": "RETRACT" if retract else "KEEP",
                                "probabilities": {"KEEP": 0.2 if retract else 0.9,
                                                  "RETRACT": 0.8 if retract else 0.1}}
            elif kind == "choice":
                # the service lists the options under `criteria` (a dict
                # name -> description, or a list)
                opts = list(q.get("criteria") or ["reject"])
                if "reject" not in opts:
                    opts.append("reject")
                answers[key] = {"choice": "reject",
                                "probabilities": {o: (1.0 if o == "reject" else 0.0)
                                                  for o in opts}}
            else:                                   # noul and anything else
                answers[key] = {"noul": 0.0}
        return answers, 0.0001, None

    monkeypatch.setattr(service, "_jev_post", _post)
    return mock
