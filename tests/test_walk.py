"""Reaching across a real corpus.

The seam between the store and the graph. Every reach in
tests/test_graph.py runs against edges the test wrote by hand, which
proves the walk works and proves nothing about whether the thing works:
a graph that was never built from a branch could fail at that seam and
the suite would stay green.

So this file reaches over notes that were actually written, by
relations that were actually derived, and then checks the two things a
hand-built graph cannot tell you:

- a relation the store holds is a relation the walk takes, and its
  weight is that weight rather than the name of the stage that found it
- holding the graph to one stage still shows what that stage left
  unconnected, which is the comparison a third stage would have to pass

product.md:

  Reaching along relations from one note to another is what makes a
  question about notes rather than about text; finding what loops back
  is what makes a cycle reportable. The same reach answers a question
  and shows the work behind it.

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.
"""

import pytest

from acciughe.index import Index, StaleGraph
from acciughe.relations import LINK, CO_OCCURRENCE, SIMILARITY


@pytest.fixture
def index(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    return Index(branch=branch, store_path=tmp_path / "state" / "graph.sqlite3")


def write(index, rel, text):
    path = index.branch / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# The three notes `chain` is about, named so a test can read the state of
# those notes without the padding that moves the counts underneath them.
CHAINED = {"a.md", "b.md", "c.md"}


def chain(index):
    """A branch where one note reaches another by each stage in turn.

    ``a.md`` states a link to ``c.md``; ``c.md`` and ``b.md`` share
    enough vocabulary to co-occur. So ``a`` reaches ``c`` by a stated
    relation and then ``b`` by an inferred one, and the two stages
    connect different notes to different things.

    Six further notes are here because the subject words are in two notes
    of three, which is above the share, so they are dropped and the edge
    goes with them. Three notes and four words in two of them is not a
    vocabulary — it is three notes. The padding is about other things
    entirely, and shares no word with a, b or c, so the notes under test
    keep the relations they are testing and only the counts change.
    """
    write(index, "a.md", "see [[c]]")
    write(index, "c.md", "the compiler pipeline reads a corpus of notes")
    write(index, "b.md", "notes about a compiler pipeline and a corpus")
    # The padding states no link and shares no word with a, b or c, so it
    # changes the counts and nothing else: no edge of any kind involves it
    # and no reach from `a` arrives there. That is what lets a test assert
    # an exact set of edges or reached notes and still be measuring the
    # three notes it is about.
    for name, text in [
        ("e.md", "printer lab duplex jams cartridge"),
        ("f.md", "printer lab duplex toner cartridge"),
        ("g.md", "allotment butt water moved committee"),
        ("h.md", "allotment fence water moved shed"),
        ("i.md", "ferry timetable harbour crossing cancelled"),
        ("j.md", "ferry harbour crossing weather cancelled"),
    ]:
        write(index, name, text)


# --- Reaching across notes that were really written -------------------

def test_a_reach_crosses_a_real_corpus(index):
    """The gap this file exists to close: a reach that was never taken
    over a real branch."""
    chain(index)
    index.refresh()

    result = index.graph().reach(seeds=["a.md"])

    assert result.reached == {"a.md", "c.md", "b.md"}


def test_a_reach_shows_the_path_through_real_notes(index):
    chain(index)
    index.refresh()

    result = index.graph().reach(seeds=["a.md"])

    assert ["a.md", "c.md", "b.md"] in result.paths


def test_a_reach_crosses_a_stated_relation(index):
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "the other end")
    index.refresh()

    assert "b.md" in index.graph().reach(seeds=["a.md"]).reached


def test_a_reach_crosses_an_inferred_relation(index):
    """Not only the relations the notes state. A co-occurrence is a
    relation too, and it is tagged as one so it never masquerades."""
    write(index, "a.md", "the compiler pipeline reads a corpus of notes")
    write(index, "b.md", "a corpus read by a pipeline that compiles")
    index.refresh()

    assert "b.md" in index.graph().reach(seeds=["a.md"]).reached


def test_a_note_with_no_relations_reaches_nothing_and_is_still_known(index):
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "the other end")
    write(index, "lonely.md", "nothing in common with anyone here")
    index.refresh()

    result = index.graph().reach(seeds=["lonely.md"])

    assert result.reached == {"lonely.md"}
    assert "lonely.md" in index.graph().notes()


def test_a_reach_says_which_notes_it_left_out(index):
    """An examination says what it left out — including notes the branch
    holds that relate to nothing at all."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "the other end")
    write(index, "lonely.md", "nothing in common with anyone here")
    index.refresh()

    result = index.graph().reach(seeds=["a.md"])

    assert result.left_out == {"lonely.md"}


def test_a_loop_in_a_real_corpus_is_a_cycle(index):
    """Finding what loops back is what makes a cycle reportable."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "see [[c]]")
    write(index, "c.md", "see [[a]]")
    index.refresh()

    assert any(set(c.notes) == {"a.md", "b.md", "c.md"} for c in index.graph().cycles())


# --- The store's rows are not the graph's rows ------------------------

def test_a_weight_is_a_number_and_not_the_stage_that_found_the_edge(index):
    """The store holds (a, b, kind, weight) and the graph holds
    (a, b, weight). Handing the store's rows over unchanged puts the
    string "link" in the weight slot, and no reach would ever say so —
    it would just quietly sort and compare nonsense."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()

    weights = [weight for _, _, weight in index.graph().edges()]

    assert weights == [1.0]
    assert all(isinstance(weight, float) for weight in weights)


def test_an_inferred_relation_carries_how_much_was_shared(index):
    """A weight is a guess at ordering, but it has to be a number that
    means something: for a co-occurrence, how many terms the two notes
    share."""
    write(index, "a.md", "compiler pipeline corpus")
    write(index, "b.md", "corpus pipeline grammar")
    index.refresh()

    assert index.graph(kind=CO_OCCURRENCE).edges() == [("a.md", "b.md", 2.0)]


# --- Holding the graph to one stage -----------------------------------

def test_a_graph_can_be_held_to_one_stage(index):
    chain(index)
    index.refresh()

    assert index.graph(kind=LINK).edges() == [("a.md", "c.md", 1.0)]


def test_a_graph_of_one_stage_still_knows_every_note(index):
    """How many notes end up connected to nothing is a question about
    the stage, so the stage's graph has to know the whole corpus to
    answer it."""
    chain(index)
    index.refresh()

    assert index.graph(kind=LINK).notes() == index.graph().notes()


def test_the_stages_disagree_about_what_is_connected(index):
    """How much of it is co-occurrence rather than relation, made
    concrete: each stage reaches a different note and leaves a different
    one stranded.

    Read over the three notes `chain` is about rather than the whole
    branch. The padding in `chain` exists only to move the counts, and it
    is stranded by both stages — a note about printers relates to nothing
    here — so counting it in would say every stage left six notes
    unconnected and hide the one note that distinguishes them.
    """
    chain(index)
    index.refresh()

    stated = index.graph(kind=LINK)
    inferred = index.graph(kind=CO_OCCURRENCE)

    assert stated.unconnected() & CHAINED == {"b.md"}
    assert inferred.unconnected() & CHAINED == {"a.md"}


def test_reaching_can_cross_from_one_stage_into_another(index):
    """A per-stage graph is an instrument for measuring a stage, not a
    substitute for the graph.

    ``a`` states a link to ``c``, and ``c`` co-occurs with ``b``, so
    the path from ``a`` to ``b`` walks a stated relation and then an
    inferred one. It belongs to no single stage, which is why neither
    stage's graph reaches ``b`` and the whole graph does. Worth being
    explicit about, because a reach measured per stage would understate
    what the graph as a whole can do.
    """
    chain(index)
    index.refresh()

    full = index.graph().reach(seeds=["a.md"])
    stated = index.graph(kind=LINK).reach(seeds=["a.md"])
    inferred = index.graph(kind=CO_OCCURRENCE).reach(seeds=["a.md"])

    assert "b.md" in full.reached
    assert "b.md" not in stated.reached | inferred.reached
    assert stated.reached | inferred.reached <= full.reached


def test_a_stage_with_no_edges_yields_a_graph_of_isolated_notes(index):
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()

    inferred = index.graph(kind=CO_OCCURRENCE)

    assert inferred.edges() == []
    assert inferred.notes() == {"a.md", "b.md"}
    assert inferred.unconnected() == {"a.md", "b.md"}


def test_the_reserved_stage_is_an_empty_graph_rather_than_an_error(index):
    """`SIMILARITY` is reserved and unimplemented, and asking for it
    is a measurement — how much a third stage would have to add to be
    worth having — not a mistake. It reports the corpus as it stands,
    which is the baseline that stage would be measured against."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()

    reserved = index.graph(kind=SIMILARITY)

    assert reserved.edges() == []
    assert reserved.unconnected() == {"a.md", "b.md"}


# --- The graph is a value, not a window --------------------------------

def test_the_graph_is_a_snapshot_that_does_not_track_a_later_refresh(index):
    """A graph is built and then examined. If it followed the store
    underneath the examination, the thing examined would not be the
    thing reported."""
    chain(index)
    index.refresh()
    snapshot = index.graph()

    write(index, "d.md", "see [[a]]")
    index.refresh()

    assert snapshot.reach(seeds=["a.md"]).reached == {"a.md", "b.md", "c.md"}
    assert "d.md" in index.graph().reach(seeds=["a.md"]).reached


def test_the_graph_follows_the_store_in_both_directions(index):
    """A relation withdrawn in the branch is gone from the next graph,
    and one added is in it. Otherwise the walk would answer from
    relations the notes no longer state."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()
    assert index.graph().edges() == [("a.md", "b.md", 1.0)]

    write(index, "a.md", "no longer mentions anything")
    index.refresh()

    assert index.graph().edges() == []


def test_a_deleted_note_is_gone_from_the_graph(index):
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()

    (index.branch / "b.md").unlink()
    index.refresh()

    assert "b.md" not in index.graph().notes()


# --- What the graph does not do ---------------------------------------

def test_reading_the_graph_does_not_refresh_the_store(index):
    """A reader that wanted the graph it was examining could not have it
    replaced underneath it, and a measurement of a stale graph would be
    impossible to take if this quietly fixed it first."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")

    empty = index.graph()

    assert empty.notes() == set()
    assert index.derivation_version() is None


def test_a_drifted_store_is_refused_before_anything_answers_from_it(index):
    """The graph is a pure read, so it will happily describe a store
    that no longer matches the branch. This is the check that stands
    between that and an answer, and it is a separate call because the
    two jobs are different: one examines a graph, the other promises
    the graph matches the notes."""
    write(index, "a.md", "see [[b]]")
    write(index, "b.md", "target")
    index.refresh()
    graph = index.graph()

    write(index, "c.md", "a note that arrived after the graph was built")

    with pytest.raises(StaleGraph):
        index.answer_from_current_graph()
    assert graph.reach(seeds=["a.md"]).reached == {"a.md", "b.md"}
