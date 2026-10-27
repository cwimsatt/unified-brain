"""GitDiffIngester plumbing on tiny_repo: commit_range (the --resume
regression), NUL-separated file names (non-ASCII paths), coverage scan."""
import pytest

import ingester
from conftest import NON_ASCII_NAME, git


@pytest.fixture(scope="module")
def ing(tiny_repo):
    return ingester.GitDiffIngester(tiny_repo)


@pytest.fixture(scope="module")
def hashes(ing):
    return ing.commit_range()


def test_commit_range_full_history_root_first(ing, hashes):
    assert len(hashes) == 3
    assert ing.parent_hash(hashes[0]) is None
    assert hashes[-1] == ing.head_hash()
    assert ing.parent_hash(hashes[2]) == hashes[1]


def test_commit_range_since_head_is_empty(ing, hashes):
    """--resume on an up-to-date repository: 0 commits, not the whole history."""
    assert ing.commit_range(since=ing.head_hash()) == []
    assert ing.commit_range(since="HEAD") == []
    assert ing.commit_range(since=ing.head_hash(), max_n=5) == []


def test_commit_range_since_root(ing, hashes):
    assert ing.commit_range(since=hashes[0]) == hashes[1:]
    assert ing.commit_range(since=hashes[0], max_n=1) == [hashes[1]]
    assert ing.commit_range(since=hashes[1]) == [hashes[2]]


def test_commit_range_max_n(ing, hashes):
    assert ing.commit_range(max_n=2) == hashes[1:]
    assert ing.commit_range(max_n=1) == [hashes[2]]


def test_commit_range_unknown_ref_raises(ing):
    with pytest.raises(RuntimeError):
        ing.commit_range(since="no-such-ref-here")


def test_changed_files_keeps_non_ascii_names(ing, hashes):
    """Without -z git quotes the name ("donn\\303\\251es.txt") and the file
    silently vanishes from the commit."""
    files = ing.changed_files(hashes[2])
    assert set(files) == {"pkg/app.py", "fixtures/blob.bin", NON_ASCII_NAME}
    assert ing.file_at_raw(hashes[2], NON_ASCII_NAME) == "café\n".encode("utf-8")
    assert ing.file_at(hashes[2], NON_ASCII_NAME) == "café\n"
    assert ing.changed_files(hashes[0]) == ["README.md", "pkg/app.py"]
    assert ing.deleted_files(hashes[2]) == []


def test_changed_files_root_commit(ing, hashes):
    assert set(ing.changed_files(hashes[1])) == {"src/lib.rs", "parts/box.scad"}


def test_changed_files_honours_skip_globs(ing, hashes, monkeypatch):
    monkeypatch.setenv("UB_SKIP_GLOBS", "*.bin")
    assert set(ing.changed_files(hashes[2])) == {"pkg/app.py", NON_ASCII_NAME}


def test_binary_file_is_not_decoded(ing, hashes):
    raw = ing.file_at_raw(hashes[2], "fixtures/blob.bin")
    assert raw.startswith(b"\x00\x01binary")
    assert ing.file_at(hashes[2], "fixtures/blob.bin") is None


def test_scan_coverage_counts_every_path(tiny_repo):
    cov = ingester.scan_coverage(tiny_repo)
    assert cov["files_total"] == 6
    assert cov["covered"] == {".py": 1, ".rs": 1, ".scad": 1}
    # .md/.txt are TEXT_EXEMPT (ingested whole-file, not coverage-counted);
    # the blob has no known magic. The non-ASCII name must be listed, not lost.
    assert NON_ASCII_NAME in cov["not_covered_files"]
    assert "README.md" in cov["not_covered_files"]
    assert cov["not_covered"][".bin [unknown binary]"] == 1
    assert cov["files_covered"] == 3 and cov["files_not_covered"] == 3


def test_deleted_file_is_reported(tiny_repo, tmp_path_factory):
    """A fourth commit in a clone deletes README.md; deleted_files sees it."""
    clone = str(tmp_path_factory.mktemp("clone"))
    git(clone, "clone", "-q", tiny_repo, ".")
    git(clone, "rm", "-q", "README.md")
    git(clone, "commit", "-q", "-m", "drop readme")
    ing2 = ingester.GitDiffIngester(clone)
    h = ing2.head_hash()
    assert ing2.deleted_files(h) == ["README.md"]
    assert ing2.changed_files(h) == []
    assert len(ing2.commit_range()) == 4
