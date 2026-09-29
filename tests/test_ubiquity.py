"""How much of the graph is held together by words that say nothing.

The gap this file closes. `test_evaluation.py` measures the graph's
density and its edge kinds, both of which stay healthy on a corpus whose
co-occurrence edges are all artefacts. Density measures how much graph
there is; this measures how much of it means anything.

That is not a hypothetical, and it is what this replaced. While the words
that mean nothing were a fixed list of English ones, a corpus in another
language kept its function words as terms: two notes that shared nothing
but grammar satisfied `_MIN_SHARED_TERMS`, every note co-occurred with
every other, and density went to 1.0 and read as a well-connected corpus.
The walk then reached everything from any seed, so the graph contributed
nothing over keyword search while looking like it contributed a great
deal. Counting the words over the branch closed that. What is left is
what counting cannot close, and this file is how much of that is left.

**No model.** The words are counted, not judged. Asking a model which
words are stopwords would put an unverifiable answer at the bottom of
the derivation, where unlike a citation nothing downstream could catch
it — every edge already derived from it would be in the store.

product.md:

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.

A measure that guesses is worse than no measure, since it is believed.
So this one reports the words, not a verdict on them: the count says how
much of the graph is noise and the list says which words, and deciding
what to do about that is not this function's to do.
"""

import pytest

from acciughe.evaluation import ubiquity
from acciughe.index import Index
from acciughe.relations import CO_OCCURRENCE


@pytest.fixture
def branch(tmp_path):
    """Six notes on two subjects, linked explicitly.

    The links are there so that "held by function words alone" can be
    told apart from "held by nothing": a corpus with no link edges at
    all would have the same co-occurrence count as a corpus held
    together entirely by grammar.
    """
    # The link sentences are worded differently each time. Phrasing them
    # all as "See [[x]]" would put "see" in every note, and the measure
    # would name it — which would be the measure working, on a linking
    # convention rather than on anything a stopword list could fix.
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text(
        "the compiler reads the notes and writes an index. See [[b]]."
    )
    (notes / "b.md").write_text(
        "the compiler writes the index from the notes. cf. [[a]]"
    )
    (notes / "c.md").write_text(
        "the allotment committee moved the water butt. See also [[d]]."
    )
    (notes / "d.md").write_text(
        "the allotment moved the butt to the far end, as in [[c]]."
    )
    (notes / "e.md").write_text(
        "the printer in the lab jams on duplex. Next: [[f]]."
    )
    (notes / "f.md").write_text(
        "the lab printer jams when duplex is on, cf. [[e]]."
    )
    return notes


@pytest.fixture
def index(branch):
    idx = Index(branch=branch, store_path=branch.parent / "graph.sqlite3")
    idx.refresh()
    return idx


def measure(branch, name="graph.sqlite3", **kwargs):
    idx = Index(branch=branch, store_path=branch.parent / name)
    idx.refresh()
    return ubiquity(idx, **kwargs)


def italian(tmp_path, links):
    """Notes sharing only Italian function words."""
    notes = tmp_path / "notes"
    notes.mkdir()
    for name, body in links.items():
        (notes / name).write_text(body)
    return notes


# Three notes is below any floor a derivation would set, and this file is
# about a measurement rather than a derivation, so it counts anyway. The
# fault it shows is the one a corpus too small to count is *exposed* to:
# with three notes, "della" is in every note and so is "centro", and the
# share alone cannot say which is the grammar.


@pytest.fixture
def half_shared(tmp_path):
    """Six notes in which half the notes' words carry no distinction.

    Two populations, built so that a count separates them and a threshold
    has to choose:

    - ``a``, ``b`` and ``e`` say the same two words, so each pair shares
      exactly ``draft`` and ``letters``. Both are in three notes of six —
      half the corpus, which no threshold can call a subject and no
      threshold can call anything else, since a subject word sits in two.
      The edges they make are real edges, held by nothing.
    - ``c`` and ``d`` share ``allotment`` and ``needs``, in two notes of
      six. That is a subject, and the edge is real.

    The links are all six, so the pairs are connected either way and the
    measure is only reading co-occurrence.
    """
    notes = tmp_path / "half"
    notes.mkdir()
    for name, body in {
        "a.md": "draft letters. see [[b]].",
        "b.md": "draft letters. cf. [[a]]",
        "c.md": "the allotment butt needs water. see [[d]].",
        "d.md": "the allotment fence needs paint. cf. [[c]]",
        "e.md": "draft letters. see [[f]].",
        "f.md": "the ferry timetable changed. cf. [[e]]",
    }.items():
        (notes / name).write_text(body)
    return notes
def italian_function_words(tmp_path, name="it"):
    """Twelve notes sharing only Italian function words.

    Twelve, because a share needs notes to be a share of: at three,
    "della" and "centro" are both in every note and the count has nothing
    to separate them with. Here the function words are in all twelve and
    each subject is in two, which is the shape a real corpus has and the
    one where a count works.
    """
    place = ("centro", "mercato", "giardino", "museo", "porto", "teatro")
    opened = ("apre", "chiude", "fiorisce", "illumina", "brilla", "splende")
    notes = tmp_path / name
    notes.mkdir()
    for n in range(12):
        (notes / f"n{n}.md").write_text(
            f"il {place[n % 6]} della citta della scuola che {opened[n % 6]}"
        )
    return notes


# --- An English corpus holds itself together ----------------------------

def test_a_well_stopworded_corpus_has_no_cooccurrence_nonsense(index):
    """"the" is dropped, so no edge is held by it alone, and the edges
    that are left are held by the subject."""
    measured = measure(index.branch)

    assert measured.share == 0.0


def test_a_well_stopworded_corpus_names_nothing(index):
    """Six English notes, and the only word they all write is "the".

    "The" is named, which is right and is the measure working: it is in
    every note and distinguishes nothing. What matters is that no *other*
    word is named. The subject words appear in two notes each and are the
    reason the edges exist, so a measure that named them would be
    measuring the subject instead of the grammar.

    It also used to name nothing at all, because there was a list and the
    list had already dropped "the" before any measurement ran. A
    diagnostic that cannot see the problem it is for is a diagnostic that
    reports the derivation agreeing with itself.
    """
    measured = measure(index.branch)

    assert dict(measured.terms) == {"the": 6}
    assert measured.held_by_these_alone == 0


# --- A corpus in another language does not -----------------------------

def test_a_corpus_in_another_language_is_now_clean(tmp_path):
    """The fault this replaced, and it closed.

    Twelve notes, every pair sharing "della", "citta", "scuola" and "che",
    each subject in two. Under a fixed English list those four were
    distinctive words, so every pair met the co-occurrence bar, the graph
    connected everything, and the walk reached everywhere and found
    nothing. Measured now: no edge is held by a word that distinguishes
    nothing, because the count dropped them.
    """
    measured = measure(italian_function_words(tmp_path), "it.sqlite3")

    assert measured.edges > 0, "the subjects still relate, and that is the point"
    assert measured.held_by_these_alone == 0
    assert measured.share == 0.0


def test_a_linking_convention_the_count_cannot_reach_is_still_named(
    half_shared,
):
    """The residue counting leaves behind, which is why this is a report
    and not a verdict.

    "see" is in three notes of six — half, where a subject word sits, and
    so no threshold here can separate the two. The derivation therefore
    keeps it, and the a-b edge is built on it, and the measure says so:
    a third of the graph is held by words that distinguish nothing.

    The two thresholds are deliberately different numbers doing different
    jobs. The derivation asks "does this word distinguish anything in
    *these* notes", where dropping a subject is the expensive mistake. The
    measure asks "would half the corpus strip this", which is a report
    about a corpus and not a decision about an edge. A measure that
    matched the derivation could only ever confirm it.
    """
    idx = Index(branch=half_shared, store_path=half_shared.parent / "edge.sqlite3")
    idx.refresh()
    measured = ubiquity(idx)

    assert dict(measured.terms).get("see") == 3
    assert idx.stopwords() == frozenset(), "the derivation kept every word"
    assert ("a.md", "b.md", 2.0) in idx.graph(kind=CO_OCCURRENCE).edges()


def test_the_words_are_named_so_the_reader_can_see_which_language(tmp_path):
    """A count of "this much of your graph is noise" is not actionable.
    "della, che, della" is: every one is a word the list should have
    known, and the words say which language it is missing."""
    named = dict(
        measure(italian_function_words(tmp_path, "named"), "it2.sqlite3").terms
    )

    assert "della" in named
    assert "che" in named
    assert named["della"] == 12


def test_a_word_only_one_note_has_is_not_reported_as_widest(tmp_path):
    """The measure is about words that distinguish nothing. A word in one
    note distinguishes as much as a word can."""
    named = dict(
        measure(italian_function_words(tmp_path, "one-note"), "it3.sqlite3").terms
    )

    assert "centro" not in named


# --- What it does not count --------------------------------------------

def test_a_link_is_never_called_noise(index):
    """Somebody wrote `[[this]]`. No vocabulary fakes that, and a graph
    held together by links is a graph somebody built."""
    measured = measure(index.branch)

    assert measured.edges == measured.edges  # co-occurrence edges only
    assert measured.held_by_these_alone == 0


def test_an_edge_held_by_one_real_word_counts_as_real(half_shared):
    """"held by these alone" is the whole claim. One genuine shared term
    out of two makes the edge real, and counting it as noise would throw
    away the reason the measure exists.

    The a-b pair share "draft" and "letters", both in half the corpus, so
    that edge is noise. The c-d pair share "allotment" and "needs", in
    two notes of six, so that edge is real. Both at once, because a claim
    that only ever sees the good cases is a claim that would also pass on
    a graph made entirely of noise.
    """
    measured = measure(half_shared, "mixed.sqlite3")

    assert 0 < measured.held_by_these_alone < measured.edges


def test_a_corpus_of_one_note_reports_nothing(tmp_path):
    """Not a division by zero, and not a claim. One note shares nothing
    with anything."""
    branch = tmp_path / "lone"
    branch.mkdir()
    (branch / "only.md").write_text("della citta della scuola")

    measured = measure(branch, "one.sqlite3")

    assert measured.share == 0.0
    assert measured.terms == []


def test_a_corpus_with_no_edges_reports_nothing(tmp_path):
    """Two notes sharing nothing at all is a corpus that cannot say
    anything about itself, and dividing by its zero edges would claim
    otherwise."""
    branch = tmp_path / "apart"
    branch.mkdir()
    (branch / "one.md").write_text("alpha beta gamma delta epsilon")
    (branch / "two.md").write_text("zeta eta theta iota kappa lambda")

    measured = measure(branch, "none.sqlite3")

    assert measured.edges == 0
    assert measured.share == 0.0


# --- The threshold is a decision, and it is visible ---------------------

def test_the_threshold_can_be_asked_for(tmp_path):
    """The share at which a word counts as saying nothing is a judgement
    about a corpus nobody has seen. It is a parameter rather than a
    constant so that a reader who disagrees can see the other number
    rather than having to trust this one."""
    # Two notes share one word, and nothing is in every note: so at half
    # the corpus that word is ubiquitous and its edge is noise, and at
    # nine tenths it is one note's word and the edge is a relation.
    branch = tmp_path / "mixed"
    branch.mkdir()
    (branch / "uno.md").write_text("il centro della citta che apre.")
    (branch / "due.md").write_text("il mercato della citta che chiude.")
    (branch / "tre.md").write_text("il giardino della scuola fiorisce.")
    (branch / "quattro.md").write_text("il museo della scuola illumina.")

    loose = measure(branch, "loose.sqlite3", share=0.5)
    strict = measure(branch, "strict.sqlite3", share=0.9)

    assert "citta" in dict(loose.terms)
    assert loose.held_by_these_alone > 0
    assert "citta" not in dict(strict.terms)
    assert strict.held_by_these_alone == 0


def test_one_ubiquitous_word_needs_two_notes_to_be_counted(tmp_path):
    """A word in every note of a two-note corpus is what half the notes
    write; a word in the only note there is does not divide by anything.
    Without the floor, a single note's every word would be reported as
    ubiquitous, which would make the list a list of the corpus."""
    branch = tmp_path / "pair"
    branch.mkdir()
    (branch / "one.md").write_text("della citta della scuola che apre.")
    (branch / "two.md").write_text("della mercato della scuola che chiude.")

    measured = measure(branch, "pair.sqlite3")

    assert all(word != "della" or count == 2
               for word, count in measured.terms)
