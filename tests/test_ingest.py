"""The ingested database: def_keys of every language, call edges, commit
order, the per-commit body copies, the embedding-pass switch and the exit
status of next_claude.py."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

import ingester
import skills_lib
from conftest import ROOT, SERVICE_ENV, NON_ASCII_NAME


def _def_keys(r):
    """{def_key: (file, symbol, line, struct)} of every stored body."""
    out = {}
    for key in r.scan_iter("code:*"):
        dk = key[len("code:"):]
        p = skills_lib.parse_def_key(dk)
        assert p is not None, dk
        out[dk] = p
    return out


def _find(keys, file, symbol, struct):
    hits = [dk for dk, (f, s, _l, t) in keys.items()
            if (f, s, t) == (file, symbol, struct)]
    assert len(hits) == 1, (file, symbol, struct, hits)
    return hits[0]


EXPECTED = [
    ("pkg/app.py", "hello", "FunctionDef"),
    ("pkg/app.py", "Greeter", "ClassDef"),
    ("pkg/app.py", "greet", "FunctionDef"),           # bare method name
    ("pkg/app.py", "<module>", "module"),
    ("README.md", "<module>", "module"),
    ("src/lib.rs", "Counter", "struct"),
    ("src/lib.rs", "Counter", "impl"),
    ("src/lib.rs", "Counter.inc", "method"),
    ("src/lib.rs", "helper", "function"),
    ("src/lib.rs", "twice", "macro"),
    ("parts/box.scad", "SIZE", "constant"),
    ("parts/box.scad", "box", "scad_module"),
    ("parts/box.scad", "area", "function"),
    ("parts/box.scad", "f", "function"),
    ("fixtures/blob.bin", "<blob>", "blob"),
    (NON_ASCII_NAME, "<module>", "module"),
]


def test_ingest_totals(ingested_db):
    t = ingested_db.totals
    assert t["commits"] == 3 and t["done"] == 3 and t["failed"] == 0
    assert t["created"] >= len(EXPECTED)
    assert t["updated"] >= 1                       # hello edited in commit 3


def test_commit_order(ingested_db):
    r = ingested_db.r
    assert r.zcard("commit_order") == 3
    expected = ingester.GitDiffIngester(ingested_db.repo).commit_range()
    assert list(r.zrange("commit_order", 0, -1)) == expected
    assert r.get("resume_point") == expected[-1]
    assert r.hget("ingest_run", "status") == "done"
    assert json.loads(r.get(f"commit_meta:{expected[0]}"))["goal"] == "python app"


def test_def_keys_cover_every_language(ingested_db):
    keys = _def_keys(ingested_db.r)
    for file, symbol, struct in EXPECTED:
        dk = _find(keys, file, symbol, struct)
        _f, _s, line, _t = keys[dk]
        assert isinstance(line, int) and line >= 1
        assert dk == f"{file}:{symbol}:{line}:{struct}"
        row = ingested_db.r.hgetall(f"code:{dk}")
        assert (row["file"], row["symbol"], row["struct_type"]) == (file, symbol, struct)
        assert row["body"]
    for file, symbol, struct in EXPECTED:
        if symbol in ("<module>", "<blob>"):
            assert keys[_find(keys, file, symbol, struct)][2] == 1
    assert not any(t == "module" and s != "<module>" for _f, s, _l, t in keys.values())


def test_active_skills_carry_the_same_def_keys(ingested_db):
    r = ingested_db.r
    sids = sorted(r.smembers("skills_active"))
    assert sids
    by_dk = {}
    for sid in sids:
        sk = json.loads(r.get(f"skill:{sid}"))
        assert not sk.get("is_composite") and sk["def_key"]
        by_dk[sk["def_key"]] = sk
    triples = {(f, s, t) for f, s, _l, t in map(skills_lib.parse_def_key, by_dk)}
    assert set(EXPECTED) <= triples


def test_call_edges(ingested_db):
    r = ingested_db.r
    keys = _def_keys(r)
    inc = _find(keys, "src/lib.rs", "Counter.inc", "method")
    edges = r.hgetall(f"calls:{inc}")
    callees = {f.rsplit(":", 1)[0] for f in edges}
    assert callees == {"helper", "println"}
    assert inc in r.smembers("called_by:helper")
    greet = _find(keys, "pkg/app.py", "greet", "FunctionDef")
    assert {f.rsplit(":", 1)[0] for f in r.hgetall(f"calls:{greet}")} == {"hello"}
    assert greet in r.smembers("called_by:hello")
    box = _find(keys, "parts/box.scad", "box", "scad_module")
    assert {f.rsplit(":", 1)[0] for f in r.hgetall(f"calls:{box}")} == {"cube"}
    f_ = _find(keys, "parts/box.scad", "f", "function")       # function_call, not module_call
    assert {f.rsplit(":", 1)[0] for f in r.hgetall(f"calls:{f_}")} == {"area"}
    assert f_ in r.smembers("called_by:area")


def test_symbol_name_index_lists_qualified_names_under_their_bare_name(ingested_db):
    """Counter.inc is reachable as `inc` (the name at a call site) and as
    Counter.inc; a bare name stays listed once under itself."""
    r = ingested_db.r
    keys = _def_keys(r)
    inc = _find(keys, "src/lib.rs", "Counter.inc", "method")
    helper = _find(keys, "src/lib.rs", "helper", "function")
    dec = lambda xs: {x.decode() if isinstance(x, bytes) else x for x in xs}
    assert f"symbol:{inc}" in dec(r.smembers("symbol_name:inc"))
    assert f"symbol:{inc}" in dec(r.smembers("symbol_name:Counter.inc"))
    assert f"symbol:{helper}" in dec(r.smembers("symbol_name:helper"))
    assert skills_lib.symbol_name_keys("Counter.inc") == ["Counter.inc", "inc"]
    assert skills_lib.symbol_name_keys("tests.near_equal_verdict") == ["tests.near_equal_verdict", "near_equal_verdict"]
    assert skills_lib.symbol_name_keys("helper") == ["helper"]
    assert skills_lib.symbol_name_keys("<module>") == ["<module>"]


def test_edit_keeps_history(ingested_db):
    """Commit 3 edits hello: the live body is the new one, the per-commit
    copy of commit 1 still holds the old text, and the skill is updated,
    not duplicated."""
    r = ingested_db.r
    hello = _find(_def_keys(r), "pkg/app.py", "hello", "FunctionDef")
    commits = r.zrange("commit_order", 0, -1)
    assert 'return "hello, world"' in r.hget(f"code:{hello}", "body")
    assert 'return "hello"' in r.hget(f"code_at:{commits[0]}:{hello}", "body")
    assert 'return "hello, world"' in r.hget(f"code_at:{commits[2]}:{hello}", "body")
    sids = [sid for sid in r.smembers("skills_active")
            if json.loads(r.get(f"skill:{sid}"))["def_key"] == hello]
    assert len(sids) == 1
    sk = json.loads(r.get(f"skill:{sids[0]}"))
    assert sk["created_commit"] == commits[0] and sk["commit_hash"] == commits[2]


def test_embed_pass_none_is_skipped(ingested_db, monkeypatch, capsys):
    import next_claude
    monkeypatch.setenv("CODE_EMBED_PASS2", "none")
    assert next_claude.embed_pass(ingested_db.r, ingested_db.rb) == 0
    assert "skipped (CODE_EMBED_PASS2=none)" in capsys.readouterr().out


def test_embed_pass_named_store_without_key_warns(ingested_db, monkeypatch, capsys):
    import next_claude
    monkeypatch.setenv("CODE_EMBED_PASS2", "voyage")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    # the run was asked to embed and wrote nothing: non-zero, so main() exits 1
    assert next_claude.embed_pass(ingested_db.r, ingested_db.rb) == 1
    out = capsys.readouterr().out
    assert "OFF (key or SDK missing)" in out and "no usable code-vector store" in out
    assert "no vectors written" in out


def test_pass2_store_names(monkeypatch):
    import next_claude
    for v in ("none", "off", "", " None "):
        monkeypatch.setenv("CODE_EMBED_PASS2", v)
        assert next_claude._pass2_store_names() == []
    monkeypatch.setenv("CODE_EMBED_PASS2", "voyage,qwen")
    assert next_claude._pass2_store_names() == ["voyage", "qwen"]
    monkeypatch.delenv("CODE_EMBED_PASS2")
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert next_claude._pass2_store_names() == []
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    assert next_claude._pass2_store_names() == ["voyage"]
    monkeypatch.setenv("DEEPINFRA_API_KEY", "k")
    assert next_claude._pass2_store_names() == ["qwen"]


def test_default_code_store():
    assert skills_lib.default_code_store("") == "none"
    assert skills_lib.default_code_store("OFF") == "none"
    assert skills_lib.default_code_store(" Voyage ") == "voyage"
    assert skills_lib.default_code_store("voyage,qwen") == "voyage,qwen"


# ── next_claude.py as a process: exit status ─────────────────────────────────

def _run(args, cwd=ROOT, code=None):
    env = {**os.environ, **SERVICE_ENV}
    cmd = [sys.executable] + (["-c", code] if code else [os.path.join(ROOT, "next_claude.py")] + args)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)


@pytest.fixture(scope="module")
def fresh_db(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cli") / "cli.db")


def test_cli_exit_0_and_resume_is_empty(tiny_repo, fresh_db):
    p = _run([tiny_repo, "--db", fresh_db, "--no-embed"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "3 commit(s) ingested" in p.stdout and "0 failed" in p.stdout
    assert "❌" not in p.stdout
    p = _run([tiny_repo, "--db", fresh_db, "--no-embed", "--resume"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "0 commit(s)" in p.stdout                 # the --resume regression
    assert "3 commit(s) ingested" not in p.stdout


def test_cli_embed_pass_none_exits_0(tiny_repo, fresh_db):
    p = _run([tiny_repo, "--db", fresh_db, "--embed-only"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "skipped (CODE_EMBED_PASS2=none)" in p.stdout


def test_cli_exit_1_when_a_commit_failed(tiny_repo, fresh_db):
    code = textwrap.dedent(f"""
        import sys
        import next_claude
        async def fake(repo_path, *, since, max_n, r, rb):
            return {{"commits": 1, "done": 0, "skipped": 0, "failed": 1,
                     "created": 0, "updated": 0, "obsoleted": 0}}
        next_claude.ingest_repo = fake
        sys.argv = ["next_claude.py", {tiny_repo!r}, "--db", {fresh_db!r}, "--no-embed"]
        next_claude.main()
    """)
    p = _run([], code=code)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "❌ 1 commit(s) and 0 embedding batch(es) failed" in p.stdout


def test_cli_exit_1_when_an_embedding_batch_failed(tiny_repo, fresh_db):
    code = textwrap.dedent(f"""
        import sys
        import next_claude
        next_claude.embed_pass = lambda r, rb: 2
        sys.argv = ["next_claude.py", {tiny_repo!r}, "--db", {fresh_db!r}, "--embed-only"]
        next_claude.main()
    """)
    p = _run([], code=code)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "❌ 0 commit(s) and 2 embedding batch(es) failed" in p.stdout


def test_cli_usage_exits_1():
    p = _run([])
    assert p.returncode == 1
    assert "unified-brain" in p.stdout
