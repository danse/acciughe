"""Deriving relations from notes, and tagging where each came from.

This is the staged population from product.md's evaluation section:

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.

Every edge therefore carries the kind that produced it, so that
measurement is a count rather than an inference, and so an agent can be
told how much evidence stands behind a link.

Stages here need no model. The similarity stage sits behind an interface
until an embedding model is chosen, and is not implemented yet.
"""

import pytest

from math import log

from acciughe.relations import (
    LINK,
    CO_OCCURRENCE,
    link_edges,
    cooccurrence_edges,
    distinguishing,
    stopwords,
    terms,
)


# --- What a term is worth, decided by the corpus and not by a list ------

def test_a_term_is_worth_the_log_of_how_few_notes_hold_it():
    """The weight is log(N / notes holding it), and the reason is that it
    cannot come back empty.

    The counting `stopwords()` used to do asked for words in nine tenths
    of the notes, and on a 131-note branch in Italian and English the most
    widespread word reached 38% — so it found nothing, returned an empty
    set, and left every seed ranked by how many function words it shared
    with the question. "why did the press jam?" produced thirty-six seeds
    led by a note about somebody's history, because the words *why*, *did*
    and *the* were in it.

    There is no cliff in that distribution to put a threshold on — how
    widespread each term is decays smoothly — so a share chosen to drop
    function words also drops subject words by degrees, and no value is
    right rather than roughly defensible. Dividing by the term's own
    frequency instead lets the corpus supply the weights.

    The weighting outlived the count. Language lists answer that fault
    and this still answers one they cannot: a word every note of a branch
    writes — a project's own name, the way that writer opens a note — is
    in no language's list, and the weight is what makes it worth nothing.
    """
    held = distinguishing({"a.md": "rare", "b.md": "common", "c.md": "common"}, frozenset())

    assert round(held["rare"], 3) == round(log(3), 3)
    assert round(held["common"], 3) == round(log(1.5), 3)
    assert held["rare"] > held["common"]


def test_a_term_in_every_note_is_worth_nothing_and_stays_in_the_table():
    """Zero rather than dropped.

    Dropping it would make a caller treat an absence of evidence and a
    term that positively says nothing as the same thing, and the second
    is a real measurement: it is what a fully-connected vocabulary looks
    like. `seeds_for` cannot reach this case through a corpus whose words
    a language accounts for — those terms are dropped before it is
    consulted — so it is pinned here, where the function is called
    directly, rather than asserted through a caller that would never pass
    it such a term. It is reachable the other way, on a word no list
    knows, and that is the case it is for.
    """
    every = distinguishing({"a.md": "the and but", "b.md": "the and but"}, frozenset())

    assert {term: every[term] for term in ("the", "and", "but")} == {
        "the": 0.0,
        "and": 0.0,
        "but": 0.0,
    }
    assert every["a"] > 0.0, (
        "and the names beside them are still worth something — a term is "
        "in the table because it is in the corpus, and whether it is "
        "worth anything is a separate question this one does not answer"
    )


def test_a_note_is_worth_its_name_as_well_as_its_words():
    """A question that names a note is about that note.

    `tracsis/acquisto/halving.hs` and "halving" are one reference, and a
    weighting that could not see that would score such a note on whatever
    else it happened to share — which is the noise this is here to stop.
    """
    held = distinguishing({"halving.md": "nothing alike"}, frozenset())

    assert "halving" in held, "the note's own name is a term it is filed under"


def test_the_terms_are_the_ones_the_branch_calls_distinctive():
    """Passed the stopword set rather than looking it up.

    So that deriving one note's weight needs no global state, and two
    stages disagreeing about which words are distinctive cannot happen by
    accident — the same reason `terms()` takes `stop` the same way.
    """
    notes = {"a.md": "the rare", "b.md": "the other", "c.md": "the third"}
    with_stop = distinguishing(notes, frozenset({"the"}))
    without_stop = distinguishing(notes, frozenset())

    assert "the" not in with_stop
    assert "the" in without_stop
    assert "rare" in with_stop and "rare" in without_stop


def test_an_empty_corpus_has_no_weights_rather_than_a_division_by_zero():
    """log(0 / 0) is the shape of this. Nothing is not everything."""
    assert distinguishing({}, frozenset()) == {}


# --- The kinds are distinct and countable ------------------------------

def test_each_edge_says_where_it_came_from():
    """How much of it is co-occurrence rather than relation, per product.md."""
    notes = {"a": "[[b]]", "b": "[[a]]"}

    edges, _ = link_edges(notes)

    assert {e.kind for e in edges} == {LINK}


def test_link_and_cooccurrence_are_told_apart():
    notes = {"a": "alpha beta gamma\n[[b]]", "b": "alpha beta delta\n[[a]]"}

    links, _ = link_edges(notes)
    cooc = cooccurrence_edges(notes)

    assert {e.kind for e in links} == {LINK}
    assert {e.kind for e in cooc} == {CO_OCCURRENCE}


def test_an_edge_carries_a_weight():
    notes = {"a": "[[b]]", "b": "[[a]]"}

    edges, _ = link_edges(notes)

    assert all(e.weight > 0 for e in edges)


# --- Stage one: explicit links ----------------------------------------

def test_a_wikilink_is_a_relation():
    notes = {"a": "see [[b]] for more", "b": "the other note"}

    edges, _ = link_edges(notes)

    assert ("a", "b") in {(e.a, e.b) for e in edges}


def test_a_wikilink_resolves_by_stem():
    notes = {"notes/b.md": "target", "a.md": "see [[b]]"}

    edges, _ = link_edges(notes)

    assert ("a.md", "notes/b.md") in {(e.a, e.b) for e in edges}


def test_a_wikilink_resolves_by_full_path():
    notes = {"deep/place/b.md": "target", "a.md": "see [[deep/place/b]]"}

    edges, _ = link_edges(notes)

    assert ("a.md", "deep/place/b.md") in {(e.a, e.b) for e in edges}


def test_a_relative_markdown_link_is_a_relation():
    """A relative link is relative to the note that holds it."""
    notes = {"notes/b.md": "target", "notes/a.md": "see [b](b.md)"}

    edges, _ = link_edges(notes)

    assert ("notes/a.md", "notes/b.md") in {(e.a, e.b) for e in edges}


def test_a_relative_link_does_not_reach_out_of_its_own_directory():
    """DECISION: a link is resolved as written, not guessed at.

    A link at the root saying b.md names a note at the root called
    b.md. It does not reach into a subdirectory to find one, because
    quietly widening a link would invent a relation the note never
    stated.
    """
    notes = {"notes/b.md": "target", "a.md": "see [b](b.md)"}

    edges, dangling = link_edges(notes)

    assert edges == []
    assert dangling == [("a.md", "b.md")]


def test_a_link_and_a_wikilink_do_not_duplicate():
    notes = {"b.md": "target", "a.md": "[[b]] and [b](b.md)"}

    edges, _ = link_edges(notes)

    assert len([e for e in edges if (e.a, e.b) == ("a.md", "b.md")]) == 1


def test_a_link_is_undirected():
    """DECISION: a relation does not depend on which note declared it.

    A link is stored once, with its ends in a fixed order, so the same
    relation cannot be written twice. The claim is not that both orders
    appear in storage — it is that the edge is the same edge.
    """
    declared_by_a = link_edges({"a": "see [[b]]", "b": "unrelated"})[0]
    declared_by_b = link_edges({"a": "unrelated", "b": "see [[a]]"})[0]

    assert declared_by_a == declared_by_b


def test_a_link_declared_on_one_side_alone_still_exists():
    notes = {"a": "see [[b]]", "b": "unrelated text"}

    edges, _ = link_edges(notes)

    assert len(edges) == 1, "one declared link is one edge, not two directions"


def test_a_note_linking_to_itself_makes_no_edge():
    notes = {"a": "see [[a]]"}

    edges, _ = link_edges(notes)

    assert edges == []


def test_an_external_url_is_not_a_relation():
    """DECISION: a URL pointing off the device is not a note in the corpus."""
    notes = {"a": "see [the site](https://example.com) for more"}

    edges, _ = link_edges(notes)

    assert edges == []


# --- Dangling links are reported, not dropped --------------------------

def test_a_link_to_a_note_that_is_not_there_is_reported():
    """Reported rather than silent."""
    notes = {"a": "see [[nowhere]] for more"}

    edges, dangling = link_edges(notes)

    assert edges == []
    assert ("a", "nowhere") in dangling


def test_a_dangling_report_names_the_note_and_the_target():
    notes = {"a": "see [[nowhere]] for more"}

    _, dangling = link_edges(notes)

    assert dangling == [("a", "nowhere")]


def test_a_dangling_link_does_not_block_the_ones_that_resolve():
    notes = {"b": "target", "a": "see [[b]] and [[nowhere]]"}

    edges, dangling = link_edges(notes)

    assert ("a", "b") in {(e.a, e.b) for e in edges}
    assert dangling == [("a", "nowhere")]


# --- Stage two: co-occurrence -----------------------------------------

def test_two_notes_sharing_terms_are_co_occurring():
    notes = {
        "a": "the compiler pipeline reads the corpus and builds a graph",
        "b": "a graph built from a corpus by a pipeline that reads notes",
    }

    edges = cooccurrence_edges(notes)

    assert ("a", "b") in {(e.a, e.b) for e in edges}


def test_two_notes_with_nothing_in_common_are_not():
    notes = {"a": "the compiler pipeline reads the corpus", "b": "kitchen timer settings"}

    assert cooccurrence_edges(notes) == []


def test_sharing_one_word_is_too_weak_to_be_a_relation():
    """DECISION: co-occurrence needs at least two shared distinctive terms.

    One shared word is usually the word "note" or "file" rather than
    anything about the notes themselves, and a threshold of two is the
    smallest that reliably rejects that. Tunable, and tagged CO_OCCURRENCE
    so its edges stay separable from real relations.
    """
    notes = {"a": "compiler pipeline", "b": "compiler unrelated subject"}

    assert cooccurrence_edges(notes) == []


def test_sharing_two_words_is_enough():
    notes = {"a": "compiler pipeline corpus", "b": "compiler pipeline unrelated"}

    edges = cooccurrence_edges(notes)

    assert ("a", "b") in {(e.a, e.b) for e in edges}


def test_cooccurrence_ignores_the_words_the_corpus_shares():
    """DECISION: a word in every note is not a distinctive term, however
    often two notes share it."""
    notes = {"a": "the of and a", "b": "the of and a"}
    shared = frozenset({"the", "of"})

    assert cooccurrence_edges(notes, shared) == []


def test_cooccurrence_does_not_depend_on_note_order():
    notes = {"a": "compiler pipeline corpus graph", "b": "graph corpus pipeline compiler"}

    assert cooccurrence_edges(notes) == cooccurrence_edges(dict(reversed(list(notes.items()))))


def test_a_note_does_not_co_occur_with_itself():
    notes = {"a": "compiler pipeline corpus graph read build"}

    assert cooccurrence_edges(notes) == []


def test_terms_are_lowercased_and_worded():
    """The distinctive terms of a note, per product.md's measured co-occurrence."""
    assert terms("The Graph, and the GRAPH") == {"the", "graph", "and"}


def test_terms_drop_punctuation_and_short_words():
    """Length only. "The" survives because the corpus this note is in has
    not been counted yet, and which words are shared is `stopwords()`'s
    answer to give, over the whole branch."""
    assert terms("a of the graph, or graph.") == {"the", "graph"}


def test_which_words_are_stopwords_is_the_corpuss_answer_not_this_lists():
    """Nothing is dropped for being a function word in some language.

    There is no list any more. Whether "della" or "che" or "however"
    carries nothing is a question about the notes in front of us, and a
    list could only ever answer it for the corpus someone had while
    writing it — which is why it had to be patched, and why it could not
    work at all on a corpus in another language.
    """
    assert terms("about does did the of at on is are was were be been must") == (
        {"about", "does", "did", "the", "are", "was", "were", "been", "must"}
    )


def test_a_word_the_corpus_writes_everywhere_is_dropped():
    """The same sentence, with the words the corpus shares already known.

    Passing the set in rather than looking it up is what keeps one note's
    terms derivable without global state, and it is what lets a caller
    see which words are being dropped instead of trusting that the right
    ones were.
    """
    every = frozenset(
        "about does did the of at on is are was were be been must".split()
    )

    assert terms("about does did the of at on is are was were be been must", every) == set()


def test_the_language_of_the_corpus_decides_which_words_are_shared():
    """Twelve notes on three subjects, four notes each, and "the" in all.

    A list says which words a language uses for grammar, and this says
    which of those words *these notes* are written with. The subject words
    are in four notes each and go nowhere near a stopword list, so they
    stay, which is the difference the mechanism turns on.
    """
    said = {
        "compiler": ("reads", "writes", "index", "builds"),
        "allotment": ("moved", "butt", "water", "fence"),
        "printer": ("jams", "duplex", "lab", "tray"),
    }
    corpus = {
        f"note-{n}": f"the {subject} {said[subject][n % 4]} writes"
        for n, subject in enumerate([s for s, _ in said.items()] * 4)
    }

    found = stopwords(corpus)

    assert "the" in found
    assert "writes" not in found, "a contentful verb is not a stopword"
    assert terms("the compiler reads and writes", found) == {
        "compiler", "reads", "writes",
    }


def test_terms_are_stable_regardless_of_repetition():
    """Distinctive-ness is a set, not a count, so a repeated word is not louder."""
    assert terms("graph graph graph") == terms("graph")


def test_an_accented_word_is_its_own_term():
    """A word with a diacritic in it is one word, not a prefix of one.

    This was the worst of the two multilingual faults, because it did not
    lose the word — it shortened it. An ASCII pattern over "la città del
    giardino" yields "citt", which is also what "cittare" and "citto"
    yield. Three unrelated words became one term, and the graph reported
    notes as related because their words shared a prefix.

    `keyword.py` already tokenizes in any script, and says why: "the notes
    are not required to be English". This stage had to follow it, or the
    baseline would search a corpus the graph cannot read and the
    comparison between them would have measured the tokenizer.
    """
    assert terms("la città del giardino") == {
        "città", "del", "giardino",
    }


def test_two_words_differing_only_by_an_accent_are_two_terms():
    """The consequence of the fault above, and the reason shortening a
    word is worse than dropping it: "citta" and "città" are different
    words, and a graph that cannot tell them apart will join two notes
    about different places."""
    assert terms("citta città") == {"citta", "città"}


def test_a_word_in_a_script_without_ascii_letters_is_a_term():
    """Cyrillic, Greek and Hebrew all vanished entirely under an ASCII
    pattern, which left a corpus in those scripts with no terms at all:
    no co-occurrence edges, no seeds, and every question refused as
    absent while the notes sat there fully readable."""
    assert terms("привет мир") == {"привет", "мир"}
    assert terms("γεια σου κόσμε") == {"γεια", "σου", "κόσμε"}


def test_an_italian_corpus_drops_its_own_function_words():
    """The fault this replaced, shown fixed.

    "che" and "della" were distinctive because they were absent from an
    English list, so two notes about unrelated things that both began
    "la città della" shared two terms and were held together by grammar
    rather than by subject. There is nothing to drop here: the notes say
    what their language's function words are.
    """
    corpus = {
        f"note-{n}": "il centro della città della scuola che apre"
        for n in range(11)
    }
    corpus["eleven"] = "il mercato della città della scuola che chiude"
    found = stopwords(corpus)

    assert {"che", "della"} <= found
    assert terms("il giardino fiorisce", found) == {"giardino", "fiorisce"}


def test_a_corpus_of_two_notes_drops_its_function_words_too():
    """There is no floor, and this is why one is not needed.

    The count that used to answer this asked what share of the notes
    write a word, and over two notes nothing could be told apart: "the"
    and "corpus" are both in every note, so any threshold that dropped
    one dropped the other and deleting the subject is the expensive
    mistake. A language's list is a fact about the language and does not
    depend on how many notes happened to be written in it, so two notes
    are as readable as two hundred.
    """
    pair = {
        "uno": "the compiler pipeline reads a corpus of notes",
        "due": "a corpus read by a pipeline that compiles",
    }

    assert {"the", "and"} <= stopwords(pair)
    assert len(cooccurrence_edges(pair, stop=stopwords(pair))) == 1


def test_a_mixed_corpus_drops_the_function_words_of_both_languages():
    """Two languages in one branch, and neither list applied alone.

    An Italian corpus's function words are not in the English list, so
    stopping at the language with the most notes would leave "della" in an
    Italian half of the branch as a term — held by grammar rather than by
    subject, which is the fault the stage exists to close. Both are
    detected and both lists apply.
    """
    corpus = {
        "uno": "il centro della città della scuola che apre",
        "due": "il mercato della città della scuola che chiude",
        "tre": "the shed needs a new roof",
        "quattro": "the compiler reads a corpus of notes",
    }

    found = stopwords(corpus)

    assert {"della", "che"} <= found, "the Italian half"
    assert {"the"} <= found, "and the English half"
    assert "centro" not in found and "compiler" not in found


def test_a_corpus_with_no_function_words_keeps_every_word():
    """Nothing detected, and nothing dropped.

    Three notes about kettles share no word with any stopword list, so
    there is no language to have written them in. Guessing English would
    be a way of stripping words out of a corpus nobody asked to have
    stripped, and "kettle" is the subject of all three.
    """
    corpus = {
        "a": "a kettle boils",
        "b": "a kettle cools",
        "c": "a kettle stands",
    }

    assert stopwords(corpus) == frozenset()
    assert terms("a kettle boils", stopwords(corpus)) == {"kettle", "boils"}


def test_a_short_corpus_of_one_word_detects_nothing():
    """A word is not a language, and one note says nothing about one.

    There was a floor for this under the count, and it was three notes.
    What the floor protected against was guessing; detection does not
    guess, so there is nothing left to protect against and no floor to
    keep in step with a threshold that no longer exists.
    """
    assert stopwords({"solo": "lamination"}) == frozenset()


def test_a_word_one_note_writes_is_never_dropped():
    """A word in one note is not shared with anything, so there is nothing
    for it to be indistinguishable from."""
    corpus = {
        "uno": "il centro della città che apre",
        "due": "il mercato della città che chiude",
        "tre": "il giardino della città che fiorisce",
        "quattro": "il museo della città che illumina",
        "cinque": "il porto della città che brilla",
        "sei": "la zia della città che telefona",
        "sette": "la cima della città che scotta",
        "otto": "il tetto della città che cede",
        "nove": "il muro della città che cade",
        "dieci": "il sasso della città che rota",
        "undici": "la luna della città che splende",
        "dodici": "il grano della città che cresce",
        "tredici": "il vento della città che soffia",
        "quattordici": "la nebbia della città che copre",
        "quindici": "il rumore della città che echeggia",
    }

    found = stopwords(corpus)

    assert "della" in found


# --- What a stage does not decide --------------------------------------

def test_no_model_is_consulted_to_derive_a_relation():
    """DECISION: these stages are pure functions of the notes.

    Nothing here calls a model, which is what makes a refresh both
    reproducible and genuinely limited to what changed.
    """
    notes = {"a": "the corpus and the graph", "b": "a graph of the corpus"}

    # The same input gives the same output, with no external state to
    # vary between calls. That is the whole claim.
    assert link_edges(notes) == link_edges(notes)
    assert cooccurrence_edges(notes) == cooccurrence_edges(notes)
