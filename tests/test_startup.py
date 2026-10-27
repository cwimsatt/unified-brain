"""The service's import-time checks, each in its own interpreter: a wrong
DB_PATH, a database without commits, an unknown JEV_PROVIDER, and the
UB_ALLOW_EMPTY_DB=1 escape hatch (the import only; uvicorn never starts)."""
import os
import subprocess
import sys

import pytest

from conftest import ROOT, SERVICE_ENV


def _import_service(db_path, **extra):
    env = {**os.environ, **SERVICE_ENV, "DB_PATH": db_path}
    env.pop("JEV_API_KEY", None)
    env.update(extra)
    return subprocess.run(
        [sys.executable, "-c", "import codebase_qa_service"],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300)


def test_missing_db_path_fails_fast(tmp_path):
    missing = str(tmp_path / "nope" / ".unified-brain.db")
    p = _import_service(missing)
    assert p.returncode != 0
    assert "does not exist" in p.stderr and missing in p.stderr
    assert "next_claude.py" in p.stderr               # the fix is named
    assert not os.path.exists(missing)                # nothing was created


def test_empty_db_fails_fast(tmp_path):
    empty = str(tmp_path / ".unified-brain.db")
    open(empty, "wb").close()
    p = _import_service(empty)
    assert p.returncode != 0
    assert "holds no commits" in p.stderr and empty in p.stderr


def test_allow_empty_db_imports(tmp_path):
    fresh = str(tmp_path / "fresh.db")
    p = _import_service(fresh, UB_ALLOW_EMPTY_DB="1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert os.path.exists(fresh)
    assert "Jev: provider=typesafe model=jev-1.13.0" in p.stdout
    assert "url=https://api.typesafe.ai/v1/systemone key=set" in p.stdout
    assert "test" not in p.stdout.split("key=")[1].split()[0]   # the key itself is never printed
    assert "code stores loaded: none" in p.stdout
    assert "QA_VECTOR_STORE=" not in p.stdout        # no warning for store none


def test_allow_empty_db_missing_key_is_reported_not_fatal(tmp_path):
    fresh = str(tmp_path / "fresh.db")
    p = _import_service(fresh, UB_ALLOW_EMPTY_DB="1", TYPESAFE_API_KEY="")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "key=MISSING (TYPESAFE_API_KEY)" in p.stdout


def test_vector_store_without_key_warns(tmp_path):
    fresh = str(tmp_path / "fresh.db")
    p = _import_service(fresh, UB_ALLOW_EMPTY_DB="1", QA_VECTOR_STORE="voyage",
                        VOYAGE_API_KEY="", DEEPINFRA_API_KEY="")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "QA_VECTOR_STORE=voyage but its key is unset" in p.stdout


def test_unknown_jev_provider_fails_at_import(tmp_path):
    fresh = str(tmp_path / "fresh.db")
    p = _import_service(fresh, UB_ALLOW_EMPTY_DB="1", JEV_PROVIDER="bogus")
    assert p.returncode != 0
    assert "JEV_PROVIDER='bogus'" in p.stderr and "ValueError" in p.stderr


@pytest.mark.parametrize("key_env,provider", [("OPENROUTER_API_KEY", "openrouter"),
                                              ("TYPESAFE_API_KEY", "typesafe")])
def test_provider_auto_detect_at_import(tmp_path, key_env, provider):
    fresh = str(tmp_path / "fresh.db")
    env = {"UB_ALLOW_EMPTY_DB": "1", "JEV_PROVIDER": "", "JEV_MODEL": "",
           "TYPESAFE_API_KEY": "", "OPENROUTER_API_KEY": "", key_env: "k"}
    p = _import_service(fresh, **env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"Jev: provider={provider} " in p.stdout
    if provider == "openrouter":
        assert "model=typesafe/jev-1.13 url=https://openrouter.ai/api/alpha/decisions" in p.stdout
