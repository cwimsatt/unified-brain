# unified-brain — handoff (2026-10-28)

A self-contained record of the work done on branch `caseyf/gallant-dijkstra-qoe9hm` so
that another session (a fresh Claude Code cloud session, a local session, or a person)
can continue without the conversation that produced it. Everything that lived only in
that session (the review findings, the measured numbers, the environment facts, the
helper scripts) is reproduced here. No secrets are in this file.

## 1. Status

| | |
|---|---|
| branch | `caseyf/gallant-dijkstra-qoe9hm`, forked from `main` at `10eaf90` |
| HEAD | `d860b25` (8 commits), pushed; `origin` in sync |
| CI | `.github/workflows/tests.yml`, Python 3.11 and 3.13: runs 1–3 green ([run 3](https://github.com/cwimsatt/unified-brain/actions/runs/40605950349)) |
| tests | 91 collected, 91 pass locally on 3.11 and 3.13 (`python -m pytest -q`) |
| diff vs main | 21 files, +2,472 / −394 |
| exercised for real | install from PyPI, grammar download, ingest of openscad-rsi (41 commits), `/api/retrieve` through the live Typesafe API (2 runs × 3 questions), the MCP server over stdio |
| not exercised | the writer (`/ask/stream`, needs an Anthropic or Fireworks key), a vanilla-agent comparison, the timeline/report pages |
| no PR opened | the user asked for pushes only |

## 2. What unified-brain is (ten lines)

`next_claude.py` walks a repository's git history with `ingester.py` (tree-sitter per
language, whole-file for the rest) and stores every version of every symbol, with call
edges, in one SQLite file `<repo>/.unified-brain.db` through `store.py` (a Redis-like API
over SQLite: kv/hashes/sets/zsets tables, an FTS5 trigram index over bodies, sqlite-vec
tables for code vectors). `skills_lib.py` is the record layer ("skills" = symbol
records, bodies per commit, edges, vectors). `codebase_qa_service.py` (FastAPI, port
8000, localhost) answers a question by: optional embedding KNN pre-filter → Jev (Typesafe's
System One model: typed `score`/`choice`/`noul` judgments, no text generation) scores every
candidate → Jev follows names through the code via FTS grep ("chains") → either a writer
LLM answers (`/ask/stream`) or the raw evidence is returned (`/api/retrieve`).
`unified_brain_mcp.py` is a thin MCP client of `/api/retrieve` with one tool, `retrieve`.
`README.md` has the long form and the file-role table.

## 3. What this branch changed, commit by commit

| commit | files | what |
|---|---|---|
| `66b3adc` | `codebase_qa_service.py`, `skills_lib.py` | Jev provider table (direct Typesafe or OpenRouter), one shared `_jev_post` helper replacing five retry loops, 502 when every judge call fails, `QA_VECTOR_STORE=none` and empty-store fallback, startup fail-fast on a missing/empty DB, `uvicorn.run(app)`, `/api/grep` regex guard and cap, defaults aligned with `env.example` |
| `414862e` | `ingester.py` | Rust and OpenSCAD extractors with call edges, `UB_SKIP_GLOBS`, `--resume` no longer re-ingests an up-to-date repo, `-z` on git name listings, dead m4/automake regex extractors removed |
| `76f50a0` | `next_claude.py`, `reembed_code_vectors.py`, `unified_brain_mcp.py`, `requirements.txt`, `.gitignore` | `CODE_EMBED_PASS2=none`, non-zero exit on failed commits/batches, honest MCP note (`info` and `log` passed through, non-JSON handled), `mcp>=1.0,<3` |
| `18b3c76` | `tests/`, `.github/workflows/tests.yml`, `pyproject.toml` | pytest suite with a synthetic 3-commit repo and a mocked Jev; CI on 3.11/3.13 |
| `8169f3f` | `README.md`, `env.example` | Typesafe provider, `none` modes, Rust/OpenSCAD, Tests, Known limitations, benchmarks wording |
| `da6ffa4` | `skills_lib.py`, `defkey_reconciler.py`, tests | qualified symbols (`Type.method`, `mod.fn`) also indexed under their bare name |
| `9702f04` | `benchmarks/openscad-rsi.md`, `README.md` | the recorded run |
| `d860b25` | several | review fixes: alias scope in drift resolution, one-off alias reindex for older databases, `JEV_CONCURRENCY` fallback for chains, `NO_GRAMMAR` diagnostics, doc corrections |

### Decisions and why

- **Provider resolution.** `JEV_PROVIDER=typesafe|openrouter` explicit; unset → `typesafe`
  when only `TYPESAFE_API_KEY` is set, else `openrouter`. Upstream users with an unchanged
  `.env` (OpenRouter key only) therefore get byte-for-byte the old requests: same URL
  `https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`, cost from
  `usage.cost`. `JEV_MODEL` is alias-mapped both ways (`typesafe/jev-1.13` ↔ `jev-1.13.0`);
  `JEV_API_BASE`, `JEV_API_KEY` override. Typesafe returns no `usage.cost`, so cost =
  `usage.input_tokens × JEV_PRICE_PER_MTOK_IN` (default 0.042 $/M).
- **`_jev_post`** retries `httpx.HTTPError`, 429, 529, ≥500 with `Retry-After` (capped 60 s)
  else `min(2**attempt, 30)`, never sleeps after the last attempt, records the last error;
  an HTTP 400 whose body mentions `max_tokens_exceeded` (Typesafe's 32k-tokens-per-question
  cap) is retried once with the header `TypeSafe-Danger-Zone-Long-Context: true`.
- **Store `none`.** `skills_lib.default_code_store(raw)`: `""`/`off`/`none` → `none`; unset →
  `qwen` if `DEEPINFRA_API_KEY`, else `voyage` if `VOYAGE_API_KEY`, else `none`. With `none`
  the KNN is skipped and Jev scores every active symbol; a configured store that returns
  nothing now also falls back to scoring everything (logged with the `--embed-only` hint)
  instead of returning 200 with no evidence.
- **OpenSCAD `module` → structureType `scad_module`.** The service treats structureType
  `module` as "whole file" (two sites), so OpenSCAD modules must not use it.
- **m4/automake stay module-only** (whole-file). The regex extractors were unreachable at
  the base commit (`MODULE_ONLY_LANGS` was checked before `NO_PARSE`) and a comment records
  module-only as a deliberate later fix, so the dead code was deleted rather than revived.
- **Bare-name alias.** Rust methods and `mod`-nested functions are stored as
  `Counter.inc` / `tests.near_equal_verdict`; call sites and the chain's definition lookup
  use the bare name, so `symbol_name:<bare>` now lists them too (register, rename and
  obsoletion keep it in sync; `skills_lib.reindex_symbol_name_aliases` upgrades an older
  database once, marker key `symbol_name_alias_v`). `_same_position_sids` filters by file
  *and* symbol so the alias cannot hand a drifted free function another symbol's history.
- **Concurrency defaults.** Select pass keeps `JEV_CONCURRENCY=128`; the chain pass reads
  `JEV_CHAIN_CONCURRENCY`, falling back to `JEV_CONCURRENCY`, default 64. `REPORT_CONCURRENCY`
  default 4 (documented value; code had 40 capped to 20). `JEV_NAMES_AS_CHOICE` default 1
  (Part 1 of `env.example`). The unused `JEV_EXPAND_MIN_SCORE` read is gone.
- **Exit codes.** `next_claude.py` exits 1 when any commit failed, any embedding batch
  failed, or a named store had no key/SDK (so "ingest ok, no vectors written" is visible);
  0 for `CODE_EMBED_PASS2=none`.
- **Startup.** `_check_db_path` exits with the path and the fix when `DB_PATH` does not
  exist or has no `commit_order` rows; `UB_ALLOW_EMPTY_DB=1` bypasses (tests use it).
- **`/api/grep`**: invalid regex → 400; `max_results` clamped to `GREP_MAX_RESULTS` (500).
- **`UB_SKIP_GLOBS`**: comma-separated fnmatch patterns on the repo-relative path or
  basename, read lazily in `is_ingestable`; default empty.

## 4. Findings of the robustness review (status)

| # | where | finding | status |
|---|---|---|---|
| 1 | `_gather_evidence` | KNN returning `[]` produced 200 with empty evidence | fixed |
| 2 | `unified_brain_mcp.py` | note said "complete evidence" for 0 bodies / failed calls | fixed |
| 3 | `_jev_select._score` | wrong key → every call fails → `[]`, still 200; 429 exhaustion unrecorded | fixed (502, `first_error`) |
| 4 | service startup | wrong `DB_PATH` silently created an empty DB | fixed |
| 5 | `commit_range` | `--resume` on an up-to-date repo re-ingested everything | fixed |
| 6 | `extract_symbols` | m4/automake regex extractors unreachable | dead code removed, docs corrected |
| 7 | `_diff_names`, `scan_coverage` | non-ASCII paths skipped (no `-z`) | fixed (`_blob_consumers` still without `-z`) |
| 8 | `next_claude.py` | exit 0 on failures | fixed |
| 9 | ingester | `.rs`/`.scad` stored whole-file | Rust and OpenSCAD extractors added |
| 10 | `__main__` | double import via `uvicorn.run("module:app")` | fixed |
| 11 | Jev client | five copies of the retry loop, hard-coded OpenRouter alpha URL | fixed |
| 12 | `/api/grep` | regex 500, unbounded results | fixed |
| 13 | defaults vs `env.example` | several mismatches | aligned |
| 14 | service | no auth, CORS `*` | deferred (README Known limitations) |
| 15 | `store.py` | non-atomic `set nx`/`incr`/`hincrbyfloat`, process-wide pipeline lock, `SQLITE_BUSY_SNAPSHOT` risk with two writers, sqlite-vec load failure crashes every connection | deferred (documented) |
| 16 | packaging | mixed pins, no lockfile, grammars downloaded at first use | documented; `pyproject.toml` added |
| 17 | `templates.py` | unsanitized model output in the pages | deferred (documented) |
| 18 | benchmarks | `xz.md` empty, `open62541.md` grading sheet blank | README wording corrected; openscad-rsi run recorded |

## 5. Environment: cloud session setup

Network access (session title bar → cloud environment menu → Edit → Network access).
Allowed domains, one per line, or tick **Allow package managers** for the first two:

| host | needed for | required |
|---|---|---|
| `pypi.org`, `files.pythonhosted.org` | pip/uv | yes |
| `github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com` | tree-sitter-language-pack downloads `parsers.json` and the grammar bundles from GitHub Releases on first use | yes |
| `api.typesafe.ai` | Jev direct | already allowed; the proxy injects the Typesafe key |
| `api.anthropic.com` | writer with `LLM_PROVIDER=anthropic` | already reachable; needs `ANTHROPIC_API_KEY` as an environment secret |
| `api.deepinfra.com` / `api.voyageai.com` | embedding pre-filter | optional; `DEEPINFRA_API_KEY` / `VOYAGE_API_KEY` |
| `openrouter.ai` | Jev via OpenRouter | optional; `OPENROUTER_API_KEY` |
| `api.fireworks.ai` | DeepSeek writer (`LLM_PROVIDER=llmbase`) | optional; `LLM_API_KEY` |

Facts that cost time to learn:

- **Network and secret changes reach new sessions only.** The running container kept the
  old policy until a new session started; `TYPESAFE_API_KEY` set as an environment secret
  was not visible in the old container either.
- **The container has no direct DNS.** `pypi.org` and `files.pythonhosted.org` sit in the
  default `no_proxy` list, so pip/uv try to go direct and fail with "Name or service not
  known". Run them with the proxy: `export NO_PROXY= no_proxy=` before `uv pip install`
  or `pip install`.
- **The proxy injects the Typesafe key** into every request to `api.typesafe.ai` and
  overrides any `Authorization` header you send (verified: a bogus bearer still gets 200).
  `.env` therefore carries `TYPESAFE_API_KEY=proxy-injected`; the value only has to be
  non-empty for the service's key check.
- **Grammar downloads through the proxy** need the Rust HTTP client to trust the proxy CA:
  `export TREE_SITTER_LANGUAGE_PACK_TLS_ROOTS=platform`. Cache:
  `~/.cache/tree-sitter-language-pack/v1.22.1/libs`.
  `TREE_SITTER_LANGUAGE_PACK_MANIFEST_URL` / `TREE_SITTER_LANGUAGE_PACK_CACHE_DIR` exist
  for mirrors and relocation.
- **`LITELLM_LOCAL_MODEL_COST_MAP=True` must be in the shell**, not only in `.env`:
  `codebase_qa_service.py` imports litellm before it reads `.env`, and litellm fetches its
  price map from `raw.githubusercontent.com` at import otherwise (it falls back to the
  bundled copy after a timeout, so this is a delay, not a failure).

Install (what was run, Python 3.11.17; 3.13.17 also passes):

```bash
cd /home/user/unified-brain
export NO_PROXY= no_proxy= TREE_SITTER_LANGUAGE_PACK_TLS_ROOTS=platform
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -r requirements.txt pytest      # every pin resolved on 2026-10-27
python - <<'PY'
import tree_sitter_language_pack as t
t.prefetch(["rust","openscad","python","bash","yaml","cmake","make","c","cpp","typescript","tsx","javascript"])
print(sorted(t.downloaded_languages()))
PY
```

The `.env` used for the test run (gitignored; copy it into a new session):

```
DB_PATH=/home/user/openscad-rsi/.unified-brain.db
JEV_PROVIDER=typesafe
JEV_MODEL=jev-1.13.0
TYPESAFE_API_KEY=proxy-injected
CODE_EMBED_PASS2=none
QA_VECTOR_STORE=none
LLM_PROVIDER=anthropic
LITELLM_LOCAL_MODEL_COST_MAP=True
UB_SKIP_GLOBS=typesafe-ai/fixtures/*,*.jsonl,*.log,*.zip,*.wasm,Cargo.lock
```

The target repository was cloned with `git clone https://github.com/cwimsatt/openscad-rsi
/home/user/openscad-rsi` and deepened with `git fetch --depth=2000 origin main`
(41 commits, not shallow afterwards). Add it to the session with the `add_repo` tool
first; the git proxy caps clones at 2 concurrent operations.

## 6. Setup on a developer machine

1. `cp env.example .env`; fill `TYPESAFE_API_KEY` (or `JEV_PROVIDER=openrouter` +
   `OPENROUTER_API_KEY`), `DB_PATH`; set `CODE_EMBED_PASS2`/`QA_VECTOR_STORE` to `qwen`
   with `DEEPINFRA_API_KEY`, or to `none` for a key-less run; `LLM_PROVIDER`/`LLM_API_KEY`
   only if you want `/ask/stream` answers. The application reads only `.env` in the
   unified-brain folder; a differently named key file (e.g. `jevClaim.env`) is not read,
   and `.gitignore` ignores `.env` only, so keep other key files out of any checkout.
2. `python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`.
3. `python next_claude.py /path/to/repo` then `python codebase_qa_service.py`.
4. MCP for Cursor (`.cursor/mcp.json`) or Claude Code (`.mcp.json`):

```json
{"mcpServers": {"unified-brain": {
  "command": "/path/to/unified-brain/.venv/bin/python",
  "args": ["/path/to/unified-brain/unified_brain_mcp.py"],
  "env": {"UNIFIED_BRAIN_URL": "http://127.0.0.1:8000"}}}}
```

## 7. How to run and verify (all commands were run in the session)

```bash
cd /home/user/unified-brain && . .venv/bin/activate && export NO_PROXY= no_proxy= LITELLM_LOCAL_MODEL_COST_MAP=True

# tests (no network, Jev mocked, ~35 s)
python -m pytest -q -p no:cacheprovider

# coverage preview: which files get symbols, which are stored whole, which are skipped
set -a; . ./.env; set +a
python ingester.py /home/user/openscad-rsi scan-coverage | python -c "import json,sys; d=json.load(sys.stdin); print(d['covered'], d['not_covered'], d['files_total'])"

# ingest (50 s for 41 commits); then --resume must report 0 commits
python next_claude.py /home/user/openscad-rsi
python next_claude.py /home/user/openscad-rsi --resume

# service in the background (stop it before deleting or re-creating the DB file)
nohup python codebase_qa_service.py > /tmp/service.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health; do sleep 1; done
grep '🧠 Jev' /tmp/service.log          # provider=typesafe model=jev-1.13.0 ... key=set

curl -s http://127.0.0.1:8000/api/symbol/near_equal_verdict | python -m json.tool | head -20
curl -s 'http://127.0.0.1:8000/api/grep?pattern=UNAVAILABLE&max_results=5' | python -m json.tool | head
curl -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8000/api/grep?pattern=%5B&regex=true'   # 400

# stop it with an exact match; `pkill -f 'python codebase_qa_service.py'` from inside a
# script that contains that text kills the script itself
pkill -xf 'python codebase_qa_service.py'
```

Retrieval with the three questions, reading the fields that matter:

```python
import httpx, time
URL = "http://127.0.0.1:8000"
for q in ["How does the verifier decide the near-equal-radius cylinder verdict?",
          "When a check is unavailable, does verification pass or fail, and where is that decided?",
          "How does the app re-verify a candidate on the frozen hash-bound snapshot?"]:
    t0 = time.time()
    d = httpx.post(f"{URL}/api/retrieve", json={"question": q}, timeout=1800).json()
    ev, info = d["evidence"], d["info"]
    print(f"{time.time()-t0:.0f}s bodies={len(ev)} chars={sum(len(e.get('code') or '') for e in ev)}")
    print(" prefilter", info["prefilter"])                      # {'store': 'none', 'ids': 0, 'fallback': False}
    print(" select", {k: info["select"][k] for k in ("candidates","scored","failed","kept","cost","first_error")})
    print(" expand", {k: v for k, v in info["expand"].items() if k != "per_seed"})
    print(" top", [f"{e['file']}:{e['symbol']}" for e in ev[:6]])
```

MCP over stdio (mcp 2.3 client; note the snake_case attribute names of mcp 2.x):

```python
import asyncio, json, os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
async def main():
    p = StdioServerParameters(command="/home/user/unified-brain/.venv/bin/python",
                              args=["/home/user/unified-brain/unified_brain_mcp.py"],
                              env={**os.environ, "UNIFIED_BRAIN_URL": "http://127.0.0.1:8000"})
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize(); print(init.server_info.name)
            print([t.name for t in (await s.list_tools()).tools])        # ['retrieve']
            res = await s.call_tool("retrieve", {"question": "What does the osrsi_pretool_deny hook block?"})
            d = json.loads(res.content[0].text); print(res.is_error, d["note"], len(d["evidence"]))
asyncio.run(main())
```

## 8. The openscad-rsi results (details in `benchmarks/openscad-rsi.md`)

Target: `cwimsatt/openscad-rsi` at `5aac724`, 41 commits, 411 tracked files (313 after the
skip globs): 87 `.rs`, 37 `.scad`, 23 `.py`, 84 `.json`, 39 `.md`.

| ingest | |
|---|---|
| wall time | 50 s (55 s on the second run), no LLM, no embeddings |
| result | 41 commits, 0 failed; skills created 2,600, updated 756, obsoleted 249; 2,351 active |
| by type | function 904, whole-file `<module>` 300, constant 279, method 258, Python FunctionDef 207, struct 138, impl 71, variable 61, namespace 36, enum 34, scad_module 29, typedef 13, ClassDef 10, prototype 5, blob 4 |
| by language | Rust 1,692, Python 380, OpenSCAD 118, JSON 76 (whole file), Markdown 25, text 18, CSG 14, TOML 8 |
| database | 68 MB, 3,016 bodies in the FTS index |
| `--resume` | 0 commits in 2 s |

| question | run | wall | bodies | chars (≈tokens) | seeds | chain added | Jev calls | Jev cost | deciding code |
|---|---|---|---|---|---|---|---|---|---|
| 1 near-equal-radius verdict | 1 | 36 s | 50 | 196k (49k) | 19 | 31 / 3 hops | 2,702 | $0.130 | 3/3 |
| 1 | 2 | 34 s | 47 | 171k (43k) | 19 | 28 / 3 hops | 2,719 | $0.129 | 3/3 |
| 2 unavailable checks pass or fail | 1 | 49 s | 120 | 393k (98k) | 60 | 60 / 3 hops | 3,086 | $0.191 | 3/4 |
| 2 | 2 | 46 s | 120 | 357k (89k) | 60 | 60 / 3 hops | 3,063 | $0.181 | 3/4 |
| 3 re-verify on the frozen snapshot | 1 | 42 s | 120 | 369k (92k) | 60 | 60 / 2 hops | 3,056 | $0.164 | 5/5 |
| 3 | 2 | 51 s | 120 | 304k (76k) | 60 | 60 / 4 hops | 3,100 | $0.160 | 5/5 |

"Deciding code" was located by hand first (grep plus the commit that introduced it):
Q1 `app/src/upgrade.rs` `tests.near_equal_verdict`, `tests.built_cylinder_verdict`,
`tests.check_near_equal`; Q2 `app/src/upgrade.rs` `tests.require_engine`/`tests.require_worker`,
`app/src/engine.rs` `validate`, and `brep-client/tests/conformance.rs` (the one miss);
Q3 `app/src/verify_seam.rs` `verify_candidate`/`recheck_fresh`, `app/src/snapshot.rs`,
`app/src/acceptance.rs` `build`, `app/src/synth.rs`. Runs agreed on 80–85 % of the bodies.
Cost model with no pre-filter: the scoring pass is one call per active symbol (2,351 ≈
$0.084, 20 s at 128 in flight); chains add $0.05–0.11. Questions 2 and 3 hit the evidence
caps (`JEV_MAX_SKILLS=60`, `JEV_EXPAND_MAX_ADD=60`), 75–100k tokens for a writer. The MCP
call: 32 s, 49 bodies, 158,748 chars of code, JSON reply 297,633 chars. 0 failed Jev calls
in about 17,700.

## 9. Review history (so nothing is re-argued)

Two multi-agent passes ran over the uncommitted diff before the commits:

1. Implementation (three agents on disjoint file groups) → tests + docs agents → three
   reviewers (service, ingester, tests/docs) → fixer. Nine findings, all minor, eight
   applied: Retry-After parse guard; the unreachable `commit_range` fallback deleted;
   `_rust_type_name` for tuple/array/fn-pointer impl types; OpenSCAD `function_call`
   edges; test env pins against a stray `.env`; MCP note for 0 candidates and for a failed
   expansion; `next_claude` exit 1 for a keyless named store; the Jev mock reading
   `criteria`. The 3.13 leg was then verified locally.
2. Final review of `10eaf90..HEAD` (regression lens, new-code lens, completeness/docs
   lens) → fixer. Found and fixed: the bare-name alias leaking into
   `skills_lib._same_position_sids` (major: a drifted free function could take a
   same-file `Type.method`'s version chain); no reindex for pre-alias databases; chain
   concurrency ignoring a lowered `JEV_CONCURRENCY`; silent whole-file fallback when a
   grammar cannot be downloaded (now `NO_GRAMMAR` on stderr, once per language);
   `.gitignore` entries merged onto one line; README's python-dotenv comment rule; the
   README overstating the openscad-rsi hit rate. Skipped as deliberate: `REPORT_CONCURRENCY`
   default 4 (documented value) and the qwen-before-voyage auto-detect order (documented
   in `env.example`). The new-code lens reproduced the extractors against all 87 `.rs`
   and 37 `.scad` files of openscad-rsi: 0 errors, 0 duplicate def_keys.

## 10. Known limitations and open items (ordered by value)

1. **MCP reply size.** One `retrieve` reply was 297,633 characters; clients with a smaller
   tool-output limit truncate it. A `max_chars` budget on the tool (truncate bodies
   tail-first with a marker) is the obvious fix; not implemented.
2. **Pre-existing ingest gap (not introduced here).** A commit that only deletes symbols
   from a file that survives, with no added or modified definition, makes
   `next_claude.py` `ingest_payload` return `None`, so the reconciler never runs for it:
   the deleted symbols stay in `skills_active`/`symbol_name` and later line drifts are
   missed. Reproduced identically on the base commit.
3. **Writer untested** here (no Anthropic/Fireworks key). `/ask/stream` and the report
   pages are unchanged in logic but their litellm paths were not run.
4. **No vanilla-agent comparison**; the README's cost claims for xz/open62541 remain
   unsubstantiated inside the repo (`benchmarks/open62541.md` is a protocol, `xz.md` empty).
5. **Deferred hardening**: no auth and CORS `*` on the localhost service; `store.py`
   non-atomic `set nx`/`incr`/`hincrbyfloat` and the process-wide pipeline lock (do not
   run the ingester against the service's database at the same time); print-based logging
   (63 prints, no `logging`); unsanitized model output in `templates.py`; FastAPI
   `on_event` and `datetime.utcnow()` deprecations; `_blob_consumers` still parses quoted
   `git grep` output (no `-z`).
6. **Upgrade note**: users with both `DEEPINFRA_API_KEY` and `VOYAGE_API_KEY` and no explicit
   store now default to qwen (documented); `JEV_NAMES_AS_CHOICE` code default is now 1.
7. **Symbol lookups by bare name** now return every `*.name` definition in the repository
   (bounded by `JEV_EXPAND_MAX_DEFS=4` in chains); for C++ users this changes what
   `_defs()` reads for common method names such as `get`.

## 11. Gotchas learned the hard way

- `pkill -f '<pattern>'` run from a script whose text contains the pattern kills the
  script (exit 144). Use `pkill -xf` or put the chain in a script file.
- Stop the service before deleting or re-creating `.unified-brain.db`: it holds an open
  file descriptor and keeps serving the deleted inode.
- The shell's working directory does not persist between tool calls in the cloud session;
  use absolute paths or `git -C`.
- `load_dotenv` is called with `override=False`: variables already in the shell win over
  `.env` (this is what lets the tests pin their environment).
- `uvicorn.run(app, ...)`, not the `"module:app"` string, when the module is `__main__`
  (the string form imports the module a second time).
- mcp 2.x renamed `FastMCP` → `MCPServer` (`mcp.server.mcpserver`) and client result
  attributes to snake_case (`server_info`, `is_error`); `unified_brain_mcp.py` handles
  both server APIs.
- tree-sitter-language-pack ≥ 1.x downloads grammars on first use; without the cache and
  without network every file of that language is stored whole. `python ingester.py <repo>
  prefetch` fetches them up front; the ingest now logs `NO_GRAMMAR` once per language.
- OpenRouter's `/api/alpha/decisions` and Typesafe's `/v1/systemone` share the request and
  answer shape (`state`, `questions{type,instructions,criteria}`, `answers`); only the URL,
  auth, model name and cost field differ.

## 12. Suggested next steps

1. MCP `max_chars` budget (small; `unified_brain_mcp.py` + a test).
2. Key-less pre-filter: rank candidates by FTS5 bm25 over the question's identifiers before
   Jev scoring (`store.grep_keys` already has the trigram index), as a cheap alternative
   to embeddings (medium).
3. Fix the delete-only-commit reconciler gap in `next_claude.py` (small, needs a test in
   `tests/test_ingest.py`).
4. Run the writer once with an Anthropic key as an environment secret (`LLM_PROVIDER=anthropic`).
5. Auth token + CORS lockdown for the service (small); `logging` instead of `print` (wide
   but mechanical); atomic `set nx`/`incr` in `store.py` via single-statement SQL (small).
6. Run the `benchmarks/open62541.md` protocol to substantiate the README's claims.
7. If contributing upstream: the CLA in `CLA.md` applies (AGPL-3.0 for local use).

## 13. Provenance

- Session: https://claude.ai/code/session_01B8SDEBdb45a81fp3T45kXa (its plan file is
  session-local; its content is folded into sections 3–4 and 10).
- Actions: [run 1](https://github.com/cwimsatt/unified-brain/actions/runs/40602474119),
  [run 2](https://github.com/cwimsatt/unified-brain/actions/runs/40603350109),
  [run 3](https://github.com/cwimsatt/unified-brain/actions/runs/40605950349).
- Measured numbers: `benchmarks/openscad-rsi.md`.
