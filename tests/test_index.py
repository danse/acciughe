"""Indexing a branch, and knowing when it has changed.

Encodes the Reading section of product.md:

  A branch is indexed before it is answered from, and re-read when it no
  longer matches. Checking is cheap — the branch is walked and compared
  with what was last read — and refreshing only redoes what changed, so
  an answer is never given from a graph that does not match the branch.

  Both happen unasked, and are reported rather than silent. What read a
  graph is recorded with a semantic version, so a reader whose
  derivation has changed rebuilds instead of answering from stale
  structure.
"""

import stat

import pytest

from acciughe.corpus import scan
from acciughe.index import Index, StaleGraph, CURRENT_DERIVATION, _fingerprint
from acciughe.relations import CO_OCCURRENCE


@pytest.fixture
def index(tmp_path):
    """An index over a branch, with its store where a real one keeps it."""
    branch = tmp_path / "notes"
    branch.mkdir()
    return Index(branch=branch, store_path=store_in(branch))


def store_in(branch):
    """The store a branch carries, at the location `cli.store_for` puts it.

    Written out here rather than imported so that these tests pin the
    location itself: a test that got the path from the code would pass
    whatever the code decided, and the decision is the thing under test.
    """
    return branch / ".acciughe" / "graph.sqlite3"


def write(branch, rel, text):
    path = branch / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- Indexing happens before answering --------------------------------

def test_a_branch_is_indexed_before_it_is_answered_from(index):
    """A branch is indexed before it is answered from."""
    write(index.branch, "a.txt", "one")

    assert index.current() is False, "an unindexed branch must not read as current"

    index.refresh()

    assert index.current() is True


def test_refreshing_an_unchanged_branch_does_nothing(index):
    """A branch that already matches does no work."""
    write(index.branch, "a.txt", "one")
    index.refresh()

    report = index.refresh()

    assert report.added == []
    assert report.changed == []
    assert report.removed == []
    assert report.read == []


# --- Checking is cheap -----------------------------------------------

def test_checking_does_not_read_the_notes(index):
    """Checking is cheap — the branch is walked and compared.

    Proven the only way that really counts: a note that cannot be opened
    still checks clean, because the check never opens it.
    """
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()

    locked = write(index.branch, "c.txt", "three")
    locked.chmod(0o000)
    try:
        assert index.current() is False, "a new file is drift, seen by stat alone"
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_a_changed_note_is_seen_without_reading_it(index):
    write(index.branch, "a.txt", "one")
    index.refresh()

    # Content changes and the size changes with it, so the recheck key
    # differs. The check must notice, and must not have read the file to
    # do so.
    path = write(index.branch, "a.txt", "one, and then some more")
    path.chmod(0o000)
    try:
        assert index.current() is False
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


# --- Drift is detected per kind ---------------------------------------

def test_a_new_file_is_drift(index):
    write(index.branch, "a.txt", "one")
    index.refresh()
    assert index.current() is True

    write(index.branch, "b.txt", "two")

    assert index.current() is False


def test_a_deleted_file_is_drift(index):
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()
    assert index.current() is True

    (index.branch / "b.txt").unlink()

    assert index.current() is False


def test_an_edited_note_is_drift(index):
    write(index.branch, "a.txt", "one")
    index.refresh()
    assert index.current() is True

    write(index.branch, "a.txt", "one edited and now longer")

    assert index.current() is False


def test_an_untouched_branch_is_not_drift(index):
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()

    assert index.current() is True


# --- Refreshing only redoes what changed ------------------------------

def test_refreshing_reads_only_what_changed(index):
    """Refreshing only redoes what changed."""
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    write(index.branch, "c.txt", "three")
    index.refresh()

    write(index.branch, "b.txt", "two, revised")

    report = index.refresh()

    assert report.read == ["b.txt"]
    assert report.added == []
    assert report.removed == []


def test_refreshing_does_not_reopen_unchanged_notes(index):
    """Refreshing only redoes what changed, proven by the filesystem.

    The unchanged note is made unreadable. A refresh that opened it would
    report it as unreadable; a refresh that skips it cannot notice, and
    the note keeps the text it was indexed with. If this assertion ever
    needs a permission dance to pass, the refresh is reading too much.
    """
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()

    write(index.branch, "b.txt", "two, revised")
    locked = index.branch / "a.txt"
    locked.chmod(0o000)
    try:
        report = index.refresh()
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert report.read == ["b.txt"]
    assert report.unreadable == []
    assert index.note_text("a.txt") == "one", (
        "the unchanged note was served from the index, not re-read"
    )
    assert index.note_text("b.txt") == "two, revised"


def test_a_deleted_note_is_dropped_from_the_index(index):
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()

    (index.branch / "a.txt").unlink()
    report = index.refresh()

    assert report.removed == ["a.txt"]
    assert index.note_ids() == {"b.txt"}


def test_a_new_note_is_added_to_the_index(index):
    write(index.branch, "a.txt", "one")
    index.refresh()

    write(index.branch, "b.txt", "two")
    report = index.refresh()

    assert report.added == ["b.txt"]
    assert index.note_ids() == {"a.txt", "b.txt"}


def test_a_moved_note_is_a_delete_and_an_add(index):
    """DECISION: a move is not tracked across.

    The tool never moves notes, so a move is one the user made. Carrying
    relations over it would assert continuity nobody asked for.
    """
    write(index.branch, "a.txt", "one")
    index.refresh()

    (index.branch / "a.txt").rename(index.branch / "b.txt")
    report = index.refresh()

    assert report.removed == ["a.txt"]
    assert report.added == ["b.txt"]


# --- Both happen unasked, and are reported ----------------------------

def test_refreshing_is_reported_rather_than_silent(index):
    """Both happen unasked, and are reported rather than silent."""
    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()
    write(index.branch, "c.txt", "three")

    report = index.refresh()

    assert report.added == ["c.txt"], "the work done is visible to the caller"


def test_an_unreadable_file_is_reported_when_the_branch_is_indexed(index):
    """A file that cannot be read as text is reported when the branch is indexed."""
    (index.branch / "blob.dat").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    write(index.branch, "a.txt", "one")

    report = index.refresh()

    assert [u.path for u in report.unreadable] == ["blob.dat"]


def test_a_note_reported_unreadable_is_not_in_the_index(index):
    (index.branch / "blob.dat").write_bytes(b"\x00\x01\x02\xff\xfe")

    index.refresh()

    assert index.note_ids() == set()


# --- Never answering from a graph that does not match -----------------

def test_answering_from_a_drifted_graph_is_refused(index):
    """An answer is never given from a graph that does not match the branch."""
    write(index.branch, "a.txt", "one")
    index.refresh()
    write(index.branch, "a.txt", "one, but now different")

    with pytest.raises(StaleGraph):
        index.answer_from_current_graph()


def test_answering_from_a_matched_graph_is_allowed(index):
    write(index.branch, "a.txt", "one")
    index.refresh()

    index.answer_from_current_graph()  # does not raise


def test_answering_from_an_unindexed_branch_is_refused(index):
    write(index.branch, "a.txt", "one")

    with pytest.raises(StaleGraph):
        index.answer_from_current_graph()


# --- What read a graph is recorded with a semantic version ------------

def test_the_index_records_the_derivation_that_wrote_it(index):
    """What read a graph is recorded with a semantic version."""
    write(index.branch, "a.txt", "one")

    index.refresh()

    assert index.derivation_version() == CURRENT_DERIVATION


def test_a_changed_derivation_rebuilds_instead_of_answering(tmp_path):
    """A reader whose derivation has changed rebuilds instead of answering
    from stale structure."""
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.txt", "one")

    old = Index(branch=branch, store_path=store)
    old.refresh()
    assert old.current() is True, "unchanged branch, unchanged derivation"

    # The derivation has moved on. Nothing about the branch changed.
    new = Index(branch=branch, store_path=store, derivation=CURRENT_DERIVATION + 1)

    assert new.current() is False, "a stale derivation is not a current graph"


def test_a_stale_derivation_cannot_be_answered_from(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.txt", "one")

    old = Index(branch=branch, store_path=store)
    old.refresh()

    new = Index(branch=branch, store_path=store, derivation=CURRENT_DERIVATION + 1)

    with pytest.raises(StaleGraph):
        new.answer_from_current_graph()


def test_a_stale_derivation_rebuilds_from_the_notes(tmp_path):
    """A rebuild is possible because the graph is derived and disposable."""
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.txt", "one")
    write(branch, "b.txt", "two")

    old = Index(branch=branch, store_path=store)
    old.refresh()

    new = Index(branch=branch, store_path=store, derivation=CURRENT_DERIVATION + 1)
    report = new.refresh()

    assert report.rebuilt is True
    assert new.note_ids() == {"a.txt", "b.txt"}
    assert new.current() is True
    assert new.derivation_version() == CURRENT_DERIVATION + 1


def test_the_store_lives_inside_the_branch_and_the_walk_never_sees_it(tmp_path):
    """A graph sits in a hidden folder in the corpus, and the corpus's own
    walk cannot see it.

    DECISION: this reverses an earlier one, which kept the store outside
    the branch for the reason a session does. product.md puts a session
    out of the branch "since what it records is note content" — and the
    store *does* record note content, the `text` column holds every note
    verbatim, so that reason reaches this file too and was the reason for
    it.

    What changed is what the location is for. Outside the branch, a graph
    is somewhere the reader has to be told about and cannot see; inside it,
    in a folder the walk prunes, a corpus carries its own structure and
    two corpora cannot be answered from each other's notes. That is what
    the reader asked for, and the spec permits it: "A graph is derived and
    disposable" constrains what a graph is, not where it may sit.

    The reason it is safe is `product.md`'s own "excluding hidden files" —
    so this test is the one that says the store is not a note, and it is
    the assertion that would fail first if the folder were not hidden.
    """
    branch = tmp_path / "notes"
    branch.mkdir()
    write(branch, "a.txt", "one")
    write(branch, "b.txt", "two")

    idx = Index(branch=branch, store_path=store_in(branch))
    idx.refresh()

    assert store_in(branch).exists()
    assert idx.note_ids() == {"a.txt", "b.txt"}, (
        "the store is in the branch and the branch is unchanged"
    )
    assert [entry.path for entry in scan(branch)] == ["a.txt", "b.txt"], (
        "the store is not a note: the walk prunes the hidden folder rather "
        "than skipping files inside it, so nothing in .acciughe is read, "
        "stat'd, or reported as unreadable"
    )


def test_a_graph_in_the_branch_is_deletable_without_cost(tmp_path):
    """product.md: "A graph is derived and disposable. It is rebuilt from
    the notes and can be deleted at any time without loss."

    The disposal has to survive the move inside the branch. If the store
    held anything the notes do not — a decision, a counter, a question
    asked — deleting it would lose it, and "disposable" would be a claim
    about a different file than the one now sitting in the corpus.
    """
    branch = tmp_path / "notes"
    branch.mkdir()
    write(branch, "a.txt", "one")
    write(branch, "b.txt", "one and two")
    store = store_in(branch)

    before = Index(branch=branch, store_path=store)
    before.refresh()
    edges = before.graph().edges()

    store.unlink()

    after = Index(branch=branch, store_path=store)
    after.refresh()

    assert after.graph().edges() == edges, (
        "the same relations, from the same notes, with the store thrown away"
    )
    assert after.current() is True


def test_a_graph_from_an_index_knows_every_note(index):
    """The graph a query runs over includes notes with no relations.

    product.md measures "how many notes end up connected to nothing", so
    the graph handed to a query must carry the whole note set, not just
    the notes an extractor happened to relate.
    """
    from acciughe.graph import Graph

    write(index.branch, "a.txt", "one")
    write(index.branch, "b.txt", "two")
    index.refresh()

    g = Graph(edges=[], notes=index.note_ids())

    assert g.notes() == {"a.txt", "b.txt"}
    assert g.unconnected() == {"a.txt", "b.txt"}, (
        "with no extractor run yet, nothing is connected"
    )


def test_deleting_the_store_loses_nothing(tmp_path):
    """A graph is derived and disposable, and can be deleted at any time."""
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.txt", "one")
    write(branch, "b.txt", "two")

    Index(branch=branch, store_path=store).refresh()
    store.unlink()

    rebuilt = Index(branch=branch, store_path=store)
    report = rebuilt.refresh()

    assert rebuilt.note_ids() == {"a.txt", "b.txt"}
    assert report.rebuilt is True


# --- What a word means is a fact about the whole branch -----------------

# Which words say nothing is read over every note, so a change to that
# set invalidates relations whose two endpoints did not move. That is the
# one case where "refreshing only redoes what changed" has to be read
# carefully rather than literally: the change is in what a word *means*,
# and the notes that changed are not the notes whose relations did.

def kettle_branch(index):
    """Two English notes related by a word English does not stop.

    `early` and `middle` share "die" and "quarry" and nothing else, so the
    edge between them rests on one word that is German's most common and
    no list of English has. It is a term here, because a branch of two
    notes does not yet look German enough for the detector to say so, and
    `german` is the note the tests below write: adding it is what settles
    the question and takes the edge away.
    """
    write(
        index.branch, "early.txt",
        "the die was cast beside the quarry and the road to the yard "
        "where the carts were kept while the timber was stacked",
    )
    write(
        index.branch, "middle.txt",
        "a die beside a quarry on the street past the depot while the rain "
        "came down and the lamps were lit",
    )


def test_a_word_set_that_moves_re_derives_edges_it_did_not_touch(index):
    """Refreshing only redoes what changed, so an answer is never given
    from a graph that does not match the branch.

    The changed note here is `german`, and the edge that must go is
    between `early` and `middle` — two notes nobody touched. What changed
    is what "die" means: on a branch of two notes it is a word no
    language claims, and with a German note beside it the word is that
    language's own and is dropped.

    A refresh that re-derived only `german` would leave the edge in the
    store and answer from a relation the branch no longer supports, which
    is the fault the sentence forbids.
    """
    kettle_branch(index)
    index.refresh()
    assert ("early.txt", "middle.txt", 2.0) in index.graph(kind=CO_OCCURRENCE).edges()

    write(
        index.branch, "german.txt",
        "der die und ist nicht mit dem haus von der mühle",
    )

    index.refresh()

    assert index.graph(kind=CO_OCCURRENCE).edges() == []


def test_an_unchanged_word_set_re_derives_nothing(index):
    """The common case, and the reason the change is noticed by
    fingerprint rather than by redoing everything.

    Adding a note to a corpus does not usually change which languages it
    is written in — here it does not even try. So the set is the same,
    and the notes whose text was not read keep the edges they had.
    """
    kettle_branch(index)
    index.refresh()

    write(index.branch, "unrelated.txt", "a note about a bicycle chain")
    report = index.refresh()

    assert report.read == ["unrelated.txt"]
    assert ("early.txt", "middle.txt", 2.0) in index.graph(kind=CO_OCCURRENCE).edges()


def test_the_index_records_the_words_that_wrote_it(index):
    """A graph is written from two things: a version and a set of words.
    Both are recorded, because both can move while the branch stands
    still, and a reader holding a graph needs to be able to see which of
    them wrote it."""
    for name, text in {
        "early.txt": "the kettle whistles loudly",
        "middle.txt": "the kettle is loud",
        "late.txt": "the kettle is old",
    }.items():
        write(index.branch, name, text)

    index.refresh()

    assert index.stop_fingerprint() == _fingerprint(index.stopwords())
    assert index.stopwords(), "a fingerprint of nothing pins nothing"


def test_a_rebuild_records_the_words_it_used_too(index):
    """A store built from nothing has no previous set to compare against,
    so it has to record its own. Without that every refresh after a
    rebuild would find the set "changed" and re-derive the whole corpus
    again — correct answers, and no cheaper than having no store at all.
    """
    for name, text in {
        "early.txt": "the kettle whistles loudly",
        "middle.txt": "the kettle is loud",
        "late.txt": "the kettle is old",
    }.items():
        write(index.branch, name, text)
    index.refresh()
    index.store_path.unlink()

    rebuilt = Index(branch=index.branch, store_path=index.store_path)
    report = rebuilt.refresh()

    assert report.rebuilt is True
    assert rebuilt.stop_fingerprint() == _fingerprint(rebuilt.stopwords())


def test_the_words_a_turn_would_use_are_the_ones_the_branch_writes_now(index):
    """The words that say nothing are derived from the notes, so reading
    them must not be cached past a refresh that changed them.

    Caching them would be the natural way to make it cheap, and it would
    be wrong: a caller holding the set from before an edit would seed a
    walk with the words a language now claims and the branch no longer
    writes as a subject.
    """
    kettle_branch(index)
    index.refresh()
    assert "die" not in index.stopwords(), (
        "two notes are not yet a branch written in German"
    )

    write(
        index.branch, "german.txt",
        "der die und ist nicht mit dem haus von der mühle",
    )
    index.refresh()

    assert "die" in index.stopwords(), "and the branch is now written in two"
