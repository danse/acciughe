"""Reaching along relations, and showing the work.

Encodes the "What can be reached" section of product.md:

  Reaching along relations from one note to another is what makes a
  question about notes rather than about text; finding what loops back
  is what makes a cycle reportable. The same reach answers a question
  and shows the work behind it.

and the constraint that structure behind an answer is inspectable, and
"an examination says what it left out."

The edge list is supplied directly. How edges come to exist is a
separate question, decided elsewhere; reach does not care.
"""

import time

import pytest

from acciughe.graph import Graph, Reach, Cycle


def graph(*edges, notes=()):
    """Build a graph from (a, b[, weight]) tuples, plus any isolated notes.

    product.md requires the graph to be measured for "how many notes end
    up connected to nothing". A note with no relations is therefore a
    note the graph still knows about, and it can only be counted if the
    graph is told it exists.
    """
    return Graph(edges, notes=notes)


def path(*notes):
    return list(notes)


# --- Reaching makes a question about notes, not text ------------------

def test_reach_follows_a_relation_to_another_note():
    """Reaching along relations from one note to another."""
    g = graph(("seed", "neighbour"))

    result = g.reach(seeds=["seed"])

    assert "neighbour" in result.reached


def test_reach_crosses_a_chain_of_relations():
    g = graph(
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
    )

    result = g.reach(seeds=["a"])

    assert result.reached == {"a", "b", "c", "d"}


def test_reach_does_not_reach_a_note_with_no_relations():
    g = graph(("a", "b"))

    result = g.reach(seeds=["a"])

    assert "c" not in result.reached


def test_reach_from_multiple_seeds_unions_them():
    g = graph(("a", "b"), ("c", "d"))

    result = g.reach(seeds=["a", "c"])

    assert result.reached == {"a", "b", "c", "d"}


def test_reach_includes_the_seed_itself():
    g = graph(("a", "b"))

    result = g.reach(seeds=["a"])

    assert "a" in result.reached


# --- The same reach answers and shows the work ------------------------

def test_reach_returns_the_path_not_just_the_destination():
    """The same reach answers a question and shows the work behind it."""
    g = graph(("a", "b"), ("b", "c"))

    result = g.reach(seeds=["a"])

    assert path("a", "b", "c") in result.paths


def test_reach_shows_a_shortest_path_for_each_note():
    g = graph(
        ("a", "long", 0.1),
        ("long", "target", 0.1),
        ("a", "target", 0.9),  # a better, shorter way
    )

    result = g.reach(seeds=["a"])

    assert path("a", "target") in result.paths, "the direct relation is the stronger path"


def test_reach_shows_how_far_each_note_was():
    g = graph(("a", "b"), ("b", "c"))

    result = g.reach(seeds=["a"])

    assert result.depth_of("a") == 0
    assert result.depth_of("b") == 1
    assert result.depth_of("c") == 2


def test_a_note_that_can_be_reached_the_long_way_is_reported_the_short_way():
    """A walk holds a note once for each depth it is arrived at.

    The long way round is real work and is not wrong, but a reach reports
    a note by its nearest path, so the question "how far is this note" has
    one answer rather than one per route. The chain is made longer than
    the short way on purpose: with the short way absent, the far end would
    be the only answer and nothing would distinguish the two.
    """
    g = graph(("a", "target"), ("a", "b"), ("b", "c"), ("c", "target"))

    result = g.reach(seeds=["a"])

    assert result.depth_of("target") == 1
    assert path("a", "target") in result.paths, (
        "the path shown is the short one, not the first the walk found"
    )
    assert result.depth_of("c") == 2, "and the rest of the chain is unaffected"


def test_two_shortest_paths_are_broken_by_name_and_not_by_the_order_relations_arrive():
    """Two paths of the same length are both correct, and one is reported.

    `target` is two relations from `a` either way round, through `m` or
    through `z`, and both are shortest. Which one gets reported is a
    choice, and a choice taken from the order the relations happen to be
    stored in would make an answer a fact about the store rather than
    about the notes: the same corpus reindexed could answer differently.
    So the tie is broken by name.

    The two graphs below are the same relations given in opposite orders.
    Each is reachable the same two ways, so neither can be reported as
    unreachable or as truncated, and the only thing that can differ
    between them is which of two equally short paths was shown.
    """
    m_first = graph(("a", "m"), ("m", "target"), ("a", "z"), ("z", "target"))
    z_first = graph(("a", "z"), ("z", "target"), ("a", "m"), ("m", "target"))

    first = m_first.reach(seeds=["a"])
    second = z_first.reach(seeds=["a"])

    assert first.paths == second.paths, (
        "the relations are the same, so the path shown must not depend on the order"
    )
    assert path("a", "m", "target") in first.paths, (
        "and the tie is broken by name, not left to whichever came first"
    )


# --- Bounded reach says it was bounded (an examination leaves out) ----

def test_reach_stops_at_a_depth_bound():
    g = graph(("a", "b"), ("b", "c"), ("c", "d"))

    result = g.reach(seeds=["a"], max_depth=1)

    assert result.reached == {"a", "b"}


def test_a_bounded_reach_says_it_was_bounded():
    """An examination says what it left out."""
    g = graph(("a", "b"), ("b", "c"), ("c", "d"))

    result = g.reach(seeds=["a"], max_depth=1)

    assert result.truncated is True


def test_an_untruncated_reach_says_it_was_not_truncated():
    g = graph(("a", "b"))

    result = g.reach(seeds=["a"], max_depth=5)

    assert result.truncated is False


def test_a_bound_that_lands_on_the_end_of_the_graph_did_not_truncate():
    """Truncated means the bound cut something off, not that it stopped.

    Here the bound is exactly the length of the chain, so the walk ran to
    the end and the note it stopped on has nothing behind it but the note
    it came from. It reached everything there was: `left_out` is empty,
    and a reach that has left nothing out cannot also have been cut short.
    The two claims are about the same examination, so only one of them can
    be true, and before this was checked the answer was the other one.
    """
    g = graph(("a", "b"))

    result = g.reach(seeds=["a"], max_depth=1)

    assert result.reached == {"a", "b"}
    assert result.left_out == set()
    assert result.truncated is False


def test_a_walk_with_no_bound_still_reaches_the_far_end_of_a_chain():
    """A walk with nothing said about depth is given a bound from the graph.

    A shortest path visits each note at most once, so no note is nearer
    than one relation fewer than there are notes, and a chain is the case
    that needs that to be true: it is the shape where the far end is
    exactly at the derived bound and a tighter one would lose it.

    No test can catch the other mistake here — an unbounded walk over a
    cycle would not fail, it would hang — so the bound is also a claim in
    the code rather than only in a test.
    """
    chain = [f"n{step}" for step in range(8)]
    g = graph(*zip(chain, chain[1:]))

    result = g.reach(seeds=["n0"])

    assert result.reached == set(chain)
    assert result.depth_of("n7") == 7


def test_a_bounded_reach_names_what_it_left_out():
    """An examination says what it left out — by note, not just 'some'."""
    g = graph(("a", "b"), ("b", "c"), ("c", "d"))

    result = g.reach(seeds=["a"], max_depth=1)

    assert set(result.left_out) == {"c", "d"}


def test_an_untruncated_reach_left_nothing_out():
    g = graph(("a", "b"))

    result = g.reach(seeds=["a"])

    assert result.left_out == set()


def test_left_out_includes_notes_no_seed_reached_at_all():
    g = graph(("a", "b"), notes=["lonely"])

    result = g.reach(seeds=["a"], max_depth=1)

    assert "lonely" in result.left_out, "the whole corpus is the frame"


def test_an_unconnected_note_is_countable():
    """How many notes end up connected to nothing, per product.md.

    The graph is measured for this, so a note with no relations has to be
    something the graph can see and count, not a gap in its knowledge.
    """
    g = graph(("a", "b"), notes=["lonely"])

    unconnected = g.unconnected()

    assert unconnected == {"lonely"}


# --- Cycles are reportable --------------------------------------------

def test_a_loop_back_is_a_cycle():
    """Finding what loops back is what makes a cycle reportable."""
    g = graph(("a", "b"), ("b", "c"), ("c", "a"))

    cycles = g.cycles()

    assert cycles, "a loop must be found"


def test_a_cycle_names_its_notes():
    g = graph(("a", "b"), ("b", "c"), ("c", "a"))

    cycles = g.cycles()

    assert any(set(c.notes) == {"a", "b", "c"} for c in cycles)


def test_a_cycle_gives_its_order():
    g = graph(("a", "b"), ("b", "c"), ("c", "a"))

    cycle = g.cycles()[0]

    assert cycle.notes[0] == cycle.notes[-1], "a cycle closes on itself"


def test_an_acyclic_graph_has_no_cycles():
    g = graph(("a", "b"), ("b", "c"))

    assert g.cycles() == []


def test_reach_over_a_cycle_terminates():
    """A cycle must not make reach loop forever."""
    g = graph(("a", "b"), ("b", "c"), ("c", "a"))

    result = g.reach(seeds=["a"])

    assert result.reached == {"a", "b", "c"}


# --- Questions that touch notes, not text -----------------------------

def test_reach_crosses_between_differently_named_notes():
    """A question about notes, reached across notes with nothing in common."""
    g = graph(("a.md", "b.md"))

    result = g.reach(seeds=["a.md"])

    assert "b.md" in result.reached


def test_a_note_related_to_itself_reaches_nothing_new():
    """A note related to itself is a relation the graph holds but cannot walk.

    It is kept, because a relation that was derived is a fact about the
    corpus and dropping it would hide one. It leads nowhere, so a walk over
    it must not step out of the note and back: the note is at distance
    zero, and a walk that made it also a relation away would report one
    note as two depths for no reason a reader could see.
    """
    g = graph(("a", "a"), ("a", "b"))

    result = g.reach(seeds=["a"])

    assert result.reached == {"a", "b"}
    assert result.depth_of("a") == 0
    assert ("a", "a", 1.0) in g.edges(), "the relation is still held"


def test_a_walk_from_no_seeds_reaches_nothing_and_leaves_the_whole_graph_out():
    """No seed means no walk, and a walk that did not happen left all of it out.

    There is nothing to start from, so the honest report is the whole
    graph in `left_out` rather than an empty reach that reads as though
    the corpus held nothing worth reaching.
    """
    g = graph(("a", "b"), ("b", "c"))

    result = g.reach(seeds=[])

    assert result.reached == set()
    assert result.paths == []
    assert set(result.left_out) == {"a", "b", "c"}
    assert result.truncated is False


# --- What the graph holds, as well as what it reaches ------------------

def test_a_graph_reports_the_relations_it_holds():
    g = graph(("b", "a"), ("c", "b", 0.5))

    assert g.edges() == [("a", "b", 1.0), ("b", "c", 0.5)]


def test_a_reported_relation_has_its_ends_in_a_fixed_order():
    """A relation has no direction, so reporting it both ways round would
    double every count made of it."""
    g = graph(("b", "a"))

    assert g.edges() == [("a", "b", 1.0)]


def test_a_relation_given_no_weight_has_the_default():
    g = graph(("a", "b"))

    assert g.edges() == [("a", "b", 1.0)]


def test_a_note_with_no_relations_reports_none():
    g = graph(("a", "b"), notes=["lonely"])

    assert g.edges() == [("a", "b", 1.0)]


def test_the_reported_relations_are_the_ones_a_reach_would_walk():
    """Read off the graph rather than kept beside it, so the two cannot
    disagree about what is in there."""
    g = graph(("a", "b"), ("b", "c", 0.5))

    touched = {note for a, b, _ in g.edges() for note in (a, b)}

    assert g.reach(seeds=["a"]).reached == touched


# --- What a walk costs ---------------------------------------------------

def test_a_walk_costs_what_the_relations_cost_and_not_their_square():
    """The one test here that asserts on cost, because nothing else can.

    A walk is the only thing in the graph that grows with the corpus, and
    how it grows is not a behaviour: 434 assertions can all hold while a
    walk gets a thousand times slower. This walk was once one recursive
    CTE, and it was correct — every assertion in this file passed with it
    — and a 2000-note graph took over seven minutes. Only timing found it.

    A chain is the shape that makes the difference legible, because its
    diameter is its own length: a walk that holds every note at every
    depth it can be arrived at emits one row per note per note, so this
    graph would be two million rows, where a walk that expands each note
    once is two thousand queries. It is the worst case rather than a
    typical one on purpose — a corpus that reads as a list is a corpus
    somebody has, and a graph whose cost is only acceptable when the
    corpus is well connected is a cost waiting for the wrong notes.

    The budget is loose by a factor of about twenty on both sides, which
    is what keeps it a guard rather than a coin toss: the walk takes
    about half a second here and the bound is ten, while the form this
    replaced did not finish in seven minutes. A slower machine makes the
    first number worse and cannot bring the second one near the bound.
    """
    names = [f"note/{step:05d}.md" for step in range(2000)]
    g = graph(*zip(names, names[1:]), notes=names)

    started = time.perf_counter()
    result = g.reach(seeds=[names[0]])
    elapsed = time.perf_counter() - started

    assert result.reached == set(names), "the whole chain is one component"
    assert elapsed < 10, (
        f"walking {len(names)} chained notes took {elapsed:.1f}s, so the walk is "
        "growing faster than the notes; a walk that holds every note at every "
        "depth is a row per note per note"
    )
