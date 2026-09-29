"""Populating the database with relations, and keeping it true.

The staged population lives in tests/test_relations.py. This file is the
seam between those pure stages and the store: edges are persisted, they
are recomputed only for the notes that changed, and the measurement
product.md asks for falls out of the ``kind`` they carry.

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.
"""

import stat

import pytest

from acciughe.index import Index, CURRENT_DERIVATION
from acciughe.relations import LINK, CO_OCCURRENCE


@pytest.fixture
def index(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    return Index(branch=branch, store_path=tmp_path / "state" / "graph.sqlite3")


def write(branch, rel, text):
    path = branch / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def pairs(index):
    """The relations in the graph, each unordered.

    An edge is undirected and stored with its ends in a fixed order, so
    a test that writes ("c", "b") when the store holds ("b", "c") is
    testing the alphabet, not the relation.
    """
    return {frozenset((a, b)) for a, b, _, _ in index.edges()}


# --- Edges are persisted ----------------------------------------------

def test_indexing_derives_edges(index):
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "the other note")

    index.refresh()

    assert index.edges() == [("a.md", "b.md", LINK, 1.0)]


def test_edges_survive_reopening_the_store(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.md", "see [[b]]")
    write(branch, "b.md", "the other note")

    Index(branch=branch, store_path=store).refresh()
    reopened = Index(branch=branch, store_path=store)

    assert reopened.edges() == [("a.md", "b.md", LINK, 1.0)]


def test_an_edge_carries_its_kind(index):
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "compiler pipeline corpus and other text")

    index.refresh()

    kinds = {e[2] for e in index.edges()}
    assert LINK in kinds


# --- Refreshing redoes only what changed -------------------------------

def test_a_new_note_gets_edges(index):
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()

    write(index.branch, "c.md", "also see [[b]]")
    index.refresh()

    assert frozenset(("c.md", "b.md")) in pairs(index)


def test_an_edited_note_has_its_edges_recomputed(index):
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()
    assert index.edges() == [("a.md", "b.md", LINK, 1.0)]

    # The link is withdrawn. The edge must go with it.
    write(index.branch, "a.md", "no longer mentions anything")
    index.refresh()

    assert index.edges() == []


def test_unchanged_notes_do_not_have_their_edges_recomputed(index):
    """Refreshing only redoes what changed, and this reaches the edges too.

    The unchanged note is made unreadable. Its edges must survive
    untouched: a population that reopened it would either fail or, worse,
    quietly drop the relations it held.
    """
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()

    write(index.branch, "c.md", "see [[b]]")
    locked = index.branch / "a.md"
    locked.chmod(0o000)
    try:
        index.refresh()
    finally:
        locked.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert frozenset(("a.md", "b.md")) in pairs(index)


def test_a_deleted_note_loses_its_edges(index):
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()
    assert len(index.edges()) == 1

    (index.branch / "a.md").unlink()
    index.refresh()

    assert index.edges() == []


def test_a_deleted_link_target_turns_the_edge_into_a_report(index):
    """A reference to a note that is not in the corpus is reported."""
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()

    (index.branch / "b.md").unlink()
    report = index.refresh()

    assert index.edges() == []
    assert report.dangling == [("a.md", "b")]


def test_a_link_is_re_derived_when_its_target_goes_away(index):
    """DECISION: link validity is a fact about the corpus, not a note.

    A link is not a function of its two notes alone the way a
    co-occurrence is: whether ``[[b]]`` reaches anything depends on b
    being in the branch. So when a target is deleted, the notes that
    pointed at it are re-derived even though they did not change, and
    the relation does not survive its target.
    """
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    index.refresh()
    assert frozenset(("a.md", "b.md")) in pairs(index)

    (index.branch / "b.md").unlink()
    index.refresh()

    assert pairs(index) == set(), "a link cannot outlive what it points at"


def test_a_link_resolves_against_the_whole_corpus_not_the_changed_notes(index):
    """A new note's link finds a target that nobody edited.

    Resolution needs every note, even though only the new one is being
    derived. Passing only the changed notes would make its link look
    like it pointed at nothing.
    """
    write(index.branch, "b.md", "target, written first and never touched again")
    index.refresh()

    write(index.branch, "c.md", "see [[b]]")
    index.refresh()

    assert frozenset(("c.md", "b.md")) in pairs(index)


# --- Rebuilding -------------------------------------------------------

def test_a_rebuild_rederives_every_edge(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.md", "see [[b]]")
    write(branch, "b.md", "target")
    Index(branch=branch, store_path=store).refresh()
    store.unlink()

    rebuilt = Index(branch=branch, store_path=store)
    rebuilt.refresh()

    assert rebuilt.edges() == [("a.md", "b.md", LINK, 1.0)]


def test_a_changed_derivation_rebuilds_the_edges(tmp_path):
    """A reader whose derivation has changed rebuilds, edges included."""
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.md", "see [[b]]")
    write(branch, "b.md", "target")
    Index(branch=branch, store_path=store).refresh()

    newer = Index(branch=branch, store_path=store, derivation=CURRENT_DERIVATION + 1)
    newer.refresh()

    assert newer.edges() == [("a.md", "b.md", LINK, 1.0)]
    assert newer.derivation_version() == CURRENT_DERIVATION + 1


def test_deleting_the_store_loses_no_relations(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    write(branch, "a.md", "see [[b]]")
    write(branch, "b.md", "target")
    Index(branch=branch, store_path=store).refresh()
    store.unlink()

    rebuilt = Index(branch=branch, store_path=store)
    rebuilt.refresh()

    assert len(rebuilt.edges()) == 1


# --- The measurement product.md asks for ------------------------------

def test_cooccurrence_and_relation_are_separable(index):
    """How much of it is co-occurrence rather than relation."""
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target note")
    write(index.branch, "c.md", "the compiler pipeline reads a corpus")
    write(index.branch, "d.md", "a corpus read by a pipeline that compiles")
    index.refresh()

    kinds = index.edges_by_kind()
    assert kinds[LINK] == 1
    assert kinds[CO_OCCURRENCE] >= 1, "the co-occurrence stage produced edges"


def test_the_kinds_never_mix(index):
    write(index.branch, "a.md", "see [[b]] compiler pipeline corpus")
    write(index.branch, "b.md", "target compiler pipeline corpus")

    index.refresh()

    for _, _, kind, _ in index.edges():
        assert kind in (LINK, CO_OCCURRENCE)


def test_notes_connected_to_nothing_are_countable_after_population(index):
    """How many notes end up connected to nothing."""
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    write(index.branch, "lonely.md", "nothing in common with anyone here")
    index.refresh()

    assert index.unconnected() == {"lonely.md"}


def test_every_note_is_accounted_for(index):
    """The frame is the whole corpus, whether or not it is connected."""
    write(index.branch, "a.md", "see [[b]]")
    write(index.branch, "b.md", "target")
    write(index.branch, "lonely.md", "alone")

    index.refresh()

    connected = {a for a, b, _, _ in index.edges()} | {b for a, b, _, _ in index.edges()}
    assert connected | index.unconnected() == index.note_ids()
