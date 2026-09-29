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
