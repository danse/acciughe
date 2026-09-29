"""What a corpus is, and what counts as a note.

Every test here names the sentence of product.md it encodes. Where a
test is *not* traceable to product.md it is marked DECISION, because
the spec is authoritative and a decision made here is not the spec
making it.
"""

import os
import stat
from pathlib import Path

import pytest

from acciughe.corpus import Unreadable, scan, read


# --- The branch -------------------------------------------------------

def test_branch_is_walked_to_its_depth(tmp_path):
    """The branch is walked to its depth."""
    (tmp_path / "a.txt").write_text("top")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "b.txt").write_text("middle")
    (tmp_path / "deep" / "deeper").mkdir()
    (tmp_path / "deep" / "deeper" / "c.txt").write_text("bottom")

    paths = {p.path for p in scan(tmp_path)}

    assert paths == {"a.txt", "deep/b.txt", "deep/deeper/c.txt"}


def test_hidden_files_are_excluded(tmp_path):
    """... excluding hidden files."""
    (tmp_path / "visible.txt").write_text("kept")
    (tmp_path / ".hidden.txt").write_text("dropped")

    paths = {p.path for p in scan(tmp_path)}

    assert paths == {"visible.txt"}


def test_hidden_directories_are_excluded_with_their_contents(tmp_path):
    """DECISION: "hidden files" is read to cover whole hidden subtrees.

    The spec excludes hidden files but does not say whether the contents
    of a hidden directory are reachable. Excluding a file and keeping
    its siblings in the same directory would be incoherent, so a hidden
    directory takes its contents with it.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("dropped")
    (tmp_path / "kept.txt").write_text("kept")

    paths = {p.path for p in scan(tmp_path)}

    assert paths == {"kept.txt"}


def test_dotfile_prefix_is_what_makes_a_file_hidden(tmp_path):
    (tmp_path / ".config").write_text("dropped")
    (tmp_path / "config").write_text("kept")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".hidden").write_text("dropped")

    assert {p.path for p in scan(tmp_path)} == {"config"}


# --- Symlinks ---------------------------------------------------------

def test_symlinked_directories_are_not_followed(tmp_path):
    """Symlinked directories are not followed."""
    outside = tmp_path.parent / "outside-corpus"
    outside.mkdir(exist_ok=True)
    (outside / "elsewhere.txt").write_text("not in the branch")
    (tmp_path / "note.txt").write_text("here")

    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    paths = {p.path for p in scan(tmp_path)}

    assert "elsewhere.txt" not in paths
    assert "link/elsewhere.txt" not in paths


def test_symlink_loop_does_not_hang(tmp_path):
    """DECISION: not following symlinked directories must make loops safe.

    The spec forbids following symlinked directories. A directory loop
    reached some other way would still be unbounded without a guard, so
    a realpath set is kept during the walk.
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.txt").write_text("x")
    (sub / "loop").symlink_to(sub, target_is_directory=True)

    paths = {p.path for p in scan(tmp_path)}

    assert paths == {"sub/inner.txt"}


# --- What is a note ---------------------------------------------------

def test_extension_does_not_decide(tmp_path):
    """It holds text files, independently from extensions."""
    (tmp_path / "notes.rst").write_text("text wearing the wrong hat")
    (tmp_path / "data.bin").write_text("also text")

    paths = {p.path for p in scan(tmp_path)}

    assert paths == {"notes.rst", "data.bin"}


def test_content_decides_what_is_read(tmp_path):
    """What is read is decided by whether the content decodes as text."""
    (tmp_path / "real.txt").write_text("decodes")
    (tmp_path / "fake.txt").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    entries = {e.path: e for e in scan(tmp_path)}
    notes, unreadable = read(tmp_path, list(entries.values()))

    assert [n.path for n in notes] == ["real.txt"]
    assert [u.path for u in unreadable] == ["fake.txt"]


def test_a_file_that_cannot_be_read_as_text_is_reported(tmp_path):
    """A file that cannot be read as text is reported when the branch is indexed."""
    (tmp_path / "good.txt").write_text("fine")
    (tmp_path / "blob.dat").write_bytes(b"\x00\x01\x02\xff\xfe")

    _, unreadable = read(tmp_path, scan(tmp_path))

    assert len(unreadable) == 1
    assert isinstance(unreadable[0], Unreadable)
    assert unreadable[0].path == "blob.dat"
    assert unreadable[0].reason  # the report says why, not merely that


def test_unreadable_is_a_report_not_a_silent_drop(tmp_path):
    """Reported, per product.md: indexing is reported rather than silent."""
    (tmp_path / "blob.dat").write_bytes(b"\x00\x01\x02\xff\xfe")

    _, unreadable = read(tmp_path, scan(tmp_path))

    # The report carries the path relative to the branch, so the user can
    # find the file without re-deriving where the walk started.
    assert unreadable[0].path == "blob.dat"
    assert unreadable[0].relative_to_branch is True


def test_valid_utf8_containing_nul_is_not_a_note(tmp_path):
    """DECISION: NUL marks bytes that were never text, even when they decode.

    Without this the NUL guard is untested, because most binary content
    fails the decoder first and never reaches it. A file of valid UTF-8
    with an embedded NUL is the case that isolates the rule: it decodes
    cleanly and is still not a note.
    """
    raw = b"before\x00after"
    # The premise, asserted: this decodes cleanly as UTF-8. The decoder
    # is not what rejects it, so the NUL rule is the only thing under
    # test here. Without that assertion a future edit to the decoder
    # could start rejecting these bytes and quietly make the test pass
    # for the wrong reason.
    assert raw.decode("utf-8")  # no exception, by construction

    (tmp_path / "looks-text.txt").write_bytes(raw)

    notes, unreadable = read(tmp_path, scan(tmp_path))

    assert notes == []
    assert [u.path for u in unreadable] == ["looks-text.txt"]
    assert "NUL" in unreadable[0].reason


def test_latin1_bytes_are_not_mistaken_for_text(tmp_path):
    """DECISION: a decode alone is not enough to mean "text".

    Most binary content decodes without error under a permissive codec.
    This test pins the stricter reading: UTF-8, strictly, and no NUL.
    """
    (tmp_path / "latin.dat").write_bytes(b"caf\xe9 not utf-8")

    _, unreadable = read(tmp_path, scan(tmp_path))

    assert [u.path for u in unreadable] == ["latin.dat"]


def test_utf8_without_a_bom_decodes(tmp_path):
    (tmp_path / "bomless.md").write_text("naïve — em dash, no BOM", encoding="utf-8")

    notes, unreadable = read(tmp_path, scan(tmp_path))

    assert [n.text for n in notes] == ["naïve — em dash, no BOM"]
    assert unreadable == []


# --- Identity ---------------------------------------------------------

def test_note_identity_does_not_change_when_content_changes(tmp_path):
    """DECISION: identity is the branch-relative path, not the content.

    The spec says a note is identified by filename and mtime and that
    content is what is read. Keying identity on content would make every
    edit a delete plus an add, losing the relation history of a note on
    each keystroke.
    """
    note = tmp_path / "a.txt"
    note.write_text("first")
    before = scan(tmp_path)[0].id

    note.write_text("second")
    after = scan(tmp_path)[0].id

    assert before == after


def test_note_identity_differs_across_paths(tmp_path):
    (tmp_path / "a.txt").write_text("identical")
    (tmp_path / "b.txt").write_text("identical")

    ids = {e.id for e in scan(tmp_path)}

    assert len(ids) == 2


def test_moving_a_note_changes_its_identity(tmp_path):
    """DECISION: a move is a delete and an add.

    The tool never moves notes itself, so a move is one the user made.
    Carrying relations across it would assert continuity the user did
    not ask for.
    """
    (tmp_path / "a.txt").write_text("x")
    before = scan(tmp_path)[0].id

    (tmp_path / "a.txt").rename(tmp_path / "b.txt")
    after = scan(tmp_path)[0].id

    assert before != after


# --- The cheap check --------------------------------------------------

def test_scan_does_not_read_contents(tmp_path):
    """DECISION: the recheck is a stat, not a read.

    product.md requires that checking is cheap because the branch is
    walked and compared, and refreshing only redoes what changed. A
    check that opened every file would not be that. Proven here by a
    file that cannot be opened.
    """
    (tmp_path / "readable.txt").write_text("fine")
    locked = tmp_path / "locked.txt"
    locked.write_text("cannot be read")
    locked.chmod(0o000)

    try:
        entries = scan(tmp_path)
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert {e.path for e in entries} == {"readable.txt", "locked.txt"}


def test_scan_reports_the_key_the_recheck_compares_on(tmp_path):
    """The filename and its modification time are used."""
    (tmp_path / "a.txt").write_text("x")
    entry = scan(tmp_path)[0]

    assert entry.path == "a.txt"
    assert entry.mtime == (tmp_path / "a.txt").stat().st_mtime


def test_scan_reports_size_alongside_mtime(tmp_path):
    """DECISION: size is part of the recheck key.

    mtime alone misses a same-size edit inside filesystem timestamp
    granularity. Adding size narrows that window; it does not close it,
    which is accepted so the check stays a stat.
    """
    (tmp_path / "a.txt").write_text("x")
    assert scan(tmp_path)[0].size == 1

    (tmp_path / "a.txt").write_text("longer content")
    assert scan(tmp_path)[0].size == len("longer content")


def test_a_directory_with_no_notes_is_not_an_error(tmp_path):
    """A file is a note; not every file is. Nothing is required."""
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / ".only-hidden").write_text("x")

    assert scan(tmp_path) == []
