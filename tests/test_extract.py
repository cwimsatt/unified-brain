"""Symbol extraction on the fixture sources: python, rust, openscad, the
callee resolver, skip globs and the m4 whole-file rule."""
import ingester
from conftest import PY_V1, RS, SCAD


def _by_name(symbols):
    """{(name, structureType): symbol}"""
    return {(s["name"], s["structureType"]): s for s in symbols}


def _calls(sym):
    return {c["name"] for c in sym.get("calls", [])}


def test_python_symbols_methods_are_bare_names():
    syms = _by_name(ingester.extract_symbols(PY_V1, "pkg/app.py"))
    assert ("hello", "FunctionDef") in syms
    assert ("Greeter", "ClassDef") in syms
    assert ("greet", "FunctionDef") in syms          # not Greeter.greet
    assert ("<module>", "module") in syms             # the import line
    assert "Greeter.greet" not in {n for n, _t in syms}
    assert _calls(syms[("greet", "FunctionDef")]) == {"hello"}
    assert syms[("hello", "FunctionDef")]["line"] < syms[("Greeter", "ClassDef")]["line"]
    for s in syms.values():
        assert "_end" not in s and "calls" in s and "refs" in s


def test_rust_resolves_to_tier1_grammar():
    lang, tier, tree = ingester.resolve_language("src/lib.rs", RS)
    assert (lang, tier) == ("rust", 1) and tree is not None
    assert ingester.LANG_BY_EXT[".rs"] == "rust"
    assert "rust" in ingester.TIER1 and "rust" in ingester._TIER1_EXTRACTORS


def test_missing_grammar_demotes_to_whole_file_and_logs_once(monkeypatch, capsys):
    def boom(lang):
        raise RuntimeError("no grammar")
    monkeypatch.setattr(ingester, "_parser", boom)
    monkeypatch.setattr(ingester, "_GRAMMAR_FAILED", set())
    assert ingester.resolve_language("a.rs", "fn main() {}")[:2] == ("text", 3)
    assert ingester.resolve_language("b.rs", "fn main() {}")[:2] == ("text", 3)
    err = capsys.readouterr().err
    assert err.count("NO_GRAMMAR") == 1 and "prefetch" in err


def test_rust_symbols():
    syms = _by_name(ingester.extract_symbols(RS, "src/lib.rs"))
    assert ("Counter", "struct") in syms
    assert ("Counter", "impl") in syms
    assert ("Counter.inc", "method") in syms
    assert ("helper", "function") in syms
    assert ("twice", "macro") in syms
    inc = syms[("Counter.inc", "method")]
    assert "fn inc" in inc["code"]
    # call_expression and macro_invocation: helper() and println!("x")
    assert _calls(inc) == {"helper", "println"}
    assert syms[("Counter", "struct")]["line"] < syms[("Counter", "impl")]["line"] \
        < inc["line"] < syms[("helper", "function")]["line"]


def test_openscad_resolves_to_tier1_grammar():
    lang, tier, tree = ingester.resolve_language("parts/box.scad", SCAD)
    assert (lang, tier) == ("openscad", 1) and tree is not None
    assert ingester.LANG_BY_EXT[".scad"] == "openscad"
    assert "openscad" in ingester.TIER1 and "openscad" in ingester._TIER1_EXTRACTORS


def test_openscad_symbols():
    syms = _by_name(ingester.extract_symbols(SCAD, "parts/box.scad"))
    assert ("SIZE", "constant") in syms
    assert ("box", "scad_module") in syms             # never 'module': that is the file
    assert ("area", "function") in syms
    assert _calls(syms[("box", "scad_module")]) == {"cube"}
    assert _calls(syms[("f", "function")]) == {"area"}       # function_call, field `function`
    assert "module" not in {t for _n, t in syms}
    assert syms[("SIZE", "constant")]["code"].startswith("SIZE = 10")   # the node ends before ';'


def test_rust_impl_for_types_without_a_name():
    """impl blocks on tuple / unit / fn-pointer types: the impl is stored by
    its trait and the members are unqualified; arrays, bounded dyn types and
    generics resolve to a bare name. Nothing with commas, parens or spaces."""
    src = (
        "trait Tr { fn req(&self); }\n"
        "impl Tr for (u32, u64) { fn req(&self) {} }\n"
        "impl Tr for () { fn req(&self) {} }\n"
        "impl Tr for fn(u32) -> u32 { fn req(&self) {} }\n"
        "impl Tr for [u8; 4] { fn req(&self) {} }\n"
        "impl Tr for dyn Fn() + Send { fn req(&self) {} }\n"
        "impl<T> Tr for Vec<T> { fn req(&self) {} }\n"
        "impl Tr for a::B<T> { fn req(&self) {} }\n"
    )
    syms = ingester.extract_symbols(src, "src/x.rs")
    names = [(s["line"], s["name"], s["structureType"]) for s in syms
             if s["structureType"] in ("impl", "method")]
    assert names == [
        (1, "Tr.req", "method"),                     # the trait's own signature
        (2, "Tr", "impl"), (2, "req", "method"),
        (3, "Tr", "impl"), (3, "req", "method"),
        (4, "Tr", "impl"), (4, "req", "method"),
        (5, "Tr for u8", "impl"), (5, "u8.req", "method"),
        (6, "Tr for Fn", "impl"), (6, "Fn.req", "method"),
        (7, "Tr for Vec", "impl"), (7, "Vec.req", "method"),
        (8, "Tr for B", "impl"), (8, "B.req", "method"),
    ]
    for s in syms:
        assert not any(ch in s["name"] for ch in ",() ;") or " for " in s["name"], s["name"]
    spans = {n for _a, _b, n, _t in ingester._scope_spans(src.encode(), "rust")}
    assert {"req", "u8.req", "Fn.req", "Vec.req", "B.req"} <= spans
    assert not any("(" in n or "," in n for n in spans)


def test_rightmost_identifier_rust_paths():
    src = b"fn f() { a::b::c(1); g::<u8>(2); x.y(3); }"
    tree = ingester._parser("rust").parse(src)
    got = {}
    for node in ingester._walk(tree.root_node):
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            got[fn.type] = ingester._rightmost_identifier(fn, src)
    assert got["scoped_identifier"] == "c"           # a::b::c
    assert got["generic_function"] == "g"            # g::<u8>
    assert got["field_expression"] == "y"            # x.y
    names = {n for _l, n, _s in ingester._extract_calls(tree, src, "rust", [src.decode()])}
    assert names == {"c", "g", "y"}
    assert ingester._rightmost_identifier(None, src) is None


def test_rust_macro_invocation_name_drops_bang():
    src = b'fn f() { println!("x"); vec![1]; }'
    tree = ingester._parser("rust").parse(src)
    names = {n for _l, n, _s in ingester._extract_calls(tree, src, "rust", [src.decode()])}
    assert names == {"println", "vec"}


def test_skip_globs(monkeypatch):
    monkeypatch.setenv("UB_SKIP_GLOBS", "*.jsonl, vendor/fixtures/*,Cargo.lock")
    assert ingester._skip_globs() == ["*.jsonl", "vendor/fixtures/*", "Cargo.lock"]
    assert not ingester.is_ingestable("data/big.jsonl")          # basename glob
    assert not ingester.is_ingestable("vendor/fixtures/a/b.bin")  # path glob
    assert not ingester.is_ingestable("rust/Cargo.lock")
    assert ingester.is_ingestable("src/lib.rs")
    assert ingester.is_ingestable("Cargo.toml")
    assert not ingester.is_ingestable("node_modules/x/index.js")  # SKIP_PATHS still apply
    monkeypatch.setenv("UB_SKIP_GLOBS", "")
    assert ingester._skip_globs() == []
    assert ingester.is_ingestable("data/big.jsonl")
    monkeypatch.delenv("UB_SKIP_GLOBS")
    assert ingester.is_ingestable("vendor/fixtures/a/b.bin")


def test_m4_is_whole_file_module():
    src = "AC_DEFUN([X_CHECK], [AC_MSG_CHECKING([x])])\nAC_INIT([x], [1.0])\n"
    lang, tier, tree = ingester.resolve_language("configure.ac", src)
    assert lang == "m4" and tree is None
    syms = ingester.extract_symbols(src, "configure.ac")
    assert [(s["name"], s["structureType"], s["line"]) for s in syms] == \
        [("<module>", "module", 1)]
    assert syms[0]["code"] == src
    assert "m4" in ingester.MODULE_ONLY_LANGS
    assert set(ingester._NO_PARSE_EXTRACTORS) == {"asm"}
    assert not hasattr(ingester, "_m4_symbols") and not hasattr(ingester, "_automake_symbols")


def test_automake_is_whole_file_module():
    src = "bin_PROGRAMS = x\nx_SOURCES = x.c\n"
    syms = ingester.extract_symbols(src, "src/Makefile.am")
    assert [(s["name"], s["structureType"]) for s in syms] == [("<module>", "module")]


def test_unknown_text_is_whole_file_module():
    syms = ingester.extract_symbols("[package]\nname = \"x\"\n", "Cargo.toml")
    assert [(s["name"], s["structureType"]) for s in syms] == [("<module>", "module")]


def test_binary_bytes_become_blob_manifest():
    syms = ingester.extract_symbols_from_bytes(b"\x00\x01abc\x00", "fixtures/blob.bin")
    assert [(s["name"], s["structureType"], s["line"]) for s in syms] == \
        [("<blob>", "blob", 1)]
    assert "sha256:" in syms[0]["code"]
