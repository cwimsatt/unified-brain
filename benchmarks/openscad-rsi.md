# openscad-rsi — a recorded run (2026-10-27)

A measured ingest-and-ask run of unified-brain on
[cwimsatt/openscad-rsi](https://github.com/cwimsatt/openscad-rsi), a Rust/Python/OpenSCAD
repository, from a cloud container. Unlike `open62541.md` (a protocol with a blank grading
sheet), this file holds numbers from an actual run. It does not compare against a
"vanilla" agent run and the writer step was not exercised (no writer key); the retrieval
is what was measured.

## Setup

| | |
|---|---|
| unified-brain | branch `caseyf/gallant-dijkstra-qoe9hm` (Typesafe provider, Rust/OpenSCAD extractors, no-embedding mode) |
| Python | 3.11.17 venv, pins from `requirements.txt` (tree-sitter-language-pack 1.22.1, sqlite-vec 0.1.9) |
| Jev | direct, `JEV_PROVIDER=typesafe`, `JEV_MODEL=jev-1.13.0`, $0.042 per million input tokens |
| Embeddings | none (`CODE_EMBED_PASS2=none`, `QA_VECTOR_STORE=none`): Jev scores every active symbol |
| Writer | none; the evidence is what an MCP client (Cursor, Claude Code) would receive |
| Skipped at ingest | `UB_SKIP_GLOBS=typesafe-ai/fixtures/*,*.jsonl,*.log,*.zip,*.wasm,Cargo.lock` (synthetic fixtures and binaries, 98 files) |
| Target | openscad-rsi at `5aac724`, 41 commits, 411 tracked files (313 after the globs): 87 `.rs`, 37 `.scad`, 23 `.py`, 84 `.json`, 39 `.md` |

## Ingest

| | |
|---|---|
| wall time | 50 s (second run 55 s) on 4 vCPUs; no LLM, no embedding pass |
| result | 41 commits, 0 failed; skills created 2,600, updated 756, obsoleted 249; 2,351 active |
| database | 68 MB `.unified-brain.db`, 3,016 distinct bodies in the FTS5 trigram index |
| `--resume` afterwards | 0 commits in 2 s (before the fix this re-ingested the whole history) |
| active skills by type | function 904, whole-file `<module>` 300, constant 279, method 258, Python FunctionDef 207, struct 138, impl 71, variable 61, namespace (`mod`) 36, enum 34, OpenSCAD module 29, typedef 13, ClassDef 10, prototype 5, blob 4 |
| by language | Rust 1,692 symbols, Python 380, OpenSCAD 118, whole-file JSON 76, Markdown 25, text 18, CSG 14, TOML 8 |

Coverage check beforehand (`python ingester.py <repo> scan-coverage`): every `.rs`, `.scad`
and `.py` file symbol-covered; `.md`/`.txt` stored whole; `.stl`/`.cqmesh` as binary manifests.

## Questions (`POST /api/retrieve`, two runs each)

The deciding code for each question was located by hand first (grep + reading the commit
that introduced it); "hit" means that definition is among the returned bodies.

| Q | run | wall | bodies | chars (≈tokens) | seeds kept | chain added | Jev calls | Jev cost | deciding code |
|---|---|---|---|---|---|---|---|---|---|
| 1 near-equal-radius verdict | 1 | 36 s | 50 | 196k (≈49k) | 19 | 31 over 3 hops | 2,702 | $0.130 | 3/3 hit |
| 1 | 2 | 34 s | 47 | 171k (≈43k) | 19 | 28 over 3 hops | 2,719 | $0.129 | 3/3 hit |
| 2 unavailable checks: pass or fail? | 1 | 49 s | 120 | 393k (≈98k) | 60 | 60 over 3 hops | 3,086 | $0.191 | 3/4 hit |
| 2 | 2 | 46 s | 120 | 357k (≈89k) | 60 | 60 over 3 hops | 3,063 | $0.181 | 3/4 hit |
| 3 re-verify on the frozen snapshot | 1 | 42 s | 120 | 369k (≈92k) | 60 | 60 over 2 hops | 3,056 | $0.164 | 5/5 hit |
| 3 | 2 | 51 s | 120 | 304k (≈76k) | 60 | 60 over 4 hops | 3,100 | $0.160 | 5/5 hit |

Questions as asked:

1. "How does the verifier decide the near-equal-radius cylinder verdict?" — evidence held
   `app/src/upgrade.rs` `tests.near_equal_verdict`, `tests.built_cylinder_verdict`,
   `tests.check_near_equal`, the trap test, the two golden verdict files and
   `verify/src/core.rs` `evaluate_section`.
2. "When a check is unavailable, does verification pass or fail, and where is that decided?"
   — evidence held `app/src/upgrade.rs` `tests.require_engine` / `tests.require_worker`
   (the `UNAVAILABLE` panics that replaced silent SKIP passes in commit `61072b2`) and
   `app/src/engine.rs` `validate`; the parallel change in
   `brep-client/tests/conformance.rs` was the one miss.
3. "How does the app re-verify a candidate on the frozen hash-bound snapshot?" — evidence
   held `app/src/verify_seam.rs` `verify_candidate` and `recheck_fresh`,
   `app/src/snapshot.rs`, `app/src/acceptance.rs` `build`, `app/src/synth.rs`,
   `app/src/fidelity.rs` `verifier_rerun`.

Observations:

- 0 failed Jev calls in 17,700 calls across the six runs; 0.6–0.8 s per call, 128 in flight.
- Without an embedding pre-filter the fixed scoring step is 2,351 calls ≈ $0.084 per question
  (about 20 s); the chains add $0.05–0.11. A pre-filter would cut the first number, not the
  second.
- Runs agree on 80–85 % of the returned bodies (40/50, 102/120, 99/120).
- Questions 2 and 3 hit the evidence caps (`JEV_MAX_SKILLS=60`, `JEV_EXPAND_MAX_ADD=60`);
  the evidence handed to a writer is then 75–100k tokens, in line with the 85–110k the
  README reports for open62541.
- `GET /api/symbol/near_equal_verdict` resolves the `mod tests`-nested Rust function after
  the bare-name alias; before it, only the qualified `tests.near_equal_verdict` did.

## MCP (stdio)

`unified_brain_mcp.py` driven by an `mcp` 2.3 client over stdio: `initialize`, `tools/list`
(`retrieve`), one `retrieve` call — see the line appended below.

Result: `initialize` → server `unified-brain`; `tools/list` → `retrieve`; `retrieve("What does the
osrsi_pretool_deny hook block, and how does it audit what it blocked?")` → 32 s, `is_error`
false, 49 bodies / 158,748 chars of code (2,351 candidates scored, 0 failed, 17 seeds kept,
$0.085), first bodies `hooks/src/bin/osrsi_pretool_deny.rs:main`,
`hooks/src/bin/osrsi_codex_pretool_deny.rs:main`, `verify/src/guard.rs:Guard`. The JSON
reply was 297,633 characters; clients with a smaller tool-output limit will need a size
budget on the tool (not implemented).
