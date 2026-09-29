"""The baseline: plain keyword search on the same branch.

product.md:

  The comparison is against plain keyword search on the same branch, a
  strong baseline for notes written in your own words.

Two things this file is careful about, because the comparison is only
worth anything if the baseline is honest:

- It is a real full-text search with stemming and BM25 ranking, not a
  tokenising toy. A strawman would make the comparison meaningless,
  which is the opposite of what the spec asks of it.
- It runs on the branch, not on the derived graph. A baseline that
  shared machinery with the thing it measures would be blind to exactly
  the failure it exists to catch.
"""

import pytest

import acciughe.keyword
from acciughe.index import Index
from acciughe.keyword import KeywordSearch


@pytest.fixture
def branch(tmp_path):
    """Twelve notes, because the filtering this baseline does is measured
    over the whole branch and a three-note branch cannot express it.

    "The" is in every note here, which is what makes
    `test_search_ignores_a_word_in_every_note` a claim rather than a
    tautology: at three notes "the" was in two of three, which is a
    majority but not the whole corpus, and the word that decides whether
    a note is found was one the count had not settled.
    """
    root = tmp_path / "notes"
    root.mkdir()
    (root / "compiler.md").write_text(
        "The compiler pipeline reads a corpus and builds a graph from it."
    )
    (root / "bread.md").write_text("A recipe for bread. Butter, flour, water, salt.")
    (root / "kitchen.md").write_text("The kitchen timer rings before the bread is done.")
    for name, text in [
        ("hall.md", "The hall ceiling needs painting before the winter."),
        ("shed.md", "The shed roof leaked where the felt had torn."),
        ("bus.md", "The bus was late again because of the roadworks."),
        ("piano.md", "The piano needs tuning twice a year, never more."),
        ("solar.md", "The solar panels made less power than the estimate."),
        ("vet.md", "The vet said the cat was overweight, not unwell."),
        ("quilt.md", "The quilt pattern came from a book at the library."),
        ("tent.md", "The tent poles bent in the wind on the ridge."),
        ("clock.md", "The clock in the hall runs four minutes fast."),
    ]:
        (root / name).write_text(text)
    return root


def paths(search, query):
    return set(search.search(query))


# --- It finds notes by the words in them -------------------------------

def test_search_finds_a_note_by_a_word_in_it(branch):
    search = KeywordSearch(branch)

    assert paths(search, "corpus") == {"compiler.md"}


def test_search_is_case_insensitive(branch):
    search = KeywordSearch(branch)

    assert paths(search, "CORPUS") == {"compiler.md"}


def test_search_finds_several_notes(branch):
    search = KeywordSearch(branch)

    assert paths(search, "bread") == {"bread.md", "kitchen.md"}


def test_search_finds_nothing_for_a_word_in_no_note(branch):
    search = KeywordSearch(branch)

    assert paths(search, "helicopter") == set()


def test_search_ignores_a_word_in_every_note(branch):
    """A stopword is not evidence. Every note has one."""
    search = KeywordSearch(branch)

    assert paths(search, "the") == set()


def test_a_query_of_nothing_but_stopwords_finds_nothing(branch):
    """Otherwise a question like "what is it" would return the corpus."""
    search = KeywordSearch(branch)

    assert paths(search, "what is it") == set()


# --- It is a strong baseline, not a tokenising toy --------------------

def test_search_ranks_the_closer_note_first(branch):
    search = KeywordSearch(branch)

    results = search.search("bread flour")

    assert results[0] == "bread.md"


def test_search_matches_a_term_against_its_stem(branch):
    """Stemming is what makes keyword search hard to beat honestly. A
    baseline that failed to match "pipelines" against "pipeline" would
    make the comparison a contest against a broken tool."""
    search = KeywordSearch(branch)

    assert paths(search, "pipelines") == {"compiler.md"}


def test_search_does_not_fold_a_latin_plural(branch):
    """DECISION: the baseline is left with the stemmer it has.

    Porter stems "corpus" and "corpora" to different tokens, so a
    search for "corpora" finds nothing. That is recorded rather than
    patched: a second rule added to catch one plural would be a
    baseline tuned to look fair, and a baseline nobody has read is not
    the honest comparison the spec asks for.
    """
    search = KeywordSearch(branch)

    assert paths(search, "corpora") == set()


def test_search_uses_or_so_a_query_would_not_demand_every_word(branch):
    """DECISION: terms are alternatives, not requirements.

    Someone typing a question into their own notes means several of the
    words, not all of them. Requiring every term would fail a query for
    reasons a reader could not see, and the failure would be read as
    the graph being useful.
    """
    search = KeywordSearch(branch)

    assert paths(search, "corpus helicopter") == {"compiler.md"}


def test_a_word_in_no_note_does_not_spoil_a_query(branch):
    search = KeywordSearch(branch)

    assert paths(search, "what does the compiler pipeline build") == {
        "compiler.md"
    }


# --- It is on the same branch, and independent of the graph -----------

def test_the_baseline_reads_the_same_whether_or_not_a_graph_exists(branch, tmp_path):
    """DECISION: the baseline shares no machinery with what it measures.

    A baseline built on the index would inherit the index's mistakes, so
    a question the graph gets wrong would be a question the baseline
    also gets wrong, and the comparison would be blind to the failure it
    exists to catch.
    """
    before = KeywordSearch(branch).search("corpus")

    Index(branch=branch, store_path=tmp_path / "graph.sqlite3").refresh()

    after = KeywordSearch(branch).search("corpus")

    assert before == after == ["compiler.md"]


def test_the_baseline_finds_what_the_graph_would_call_unconnected(branch, tmp_path):
    """The whole point of the comparison: the two disagree, and the
    baseline reaches notes the graph has joined to nothing."""
    index = Index(branch=branch, store_path=tmp_path / "graph.sqlite3")
    index.refresh()
    search = KeywordSearch(branch)

    found = set(search.search("bread"))

    assert found == {"bread.md", "kitchen.md"}
    assert found <= index.unconnected()


def test_a_note_added_later_is_found_without_reindexing_by_hand(branch):
    """The caller is not asked to notice that the branch moved."""
    search = KeywordSearch(branch)
    assert paths(search, "helicopter") == set()

    (branch / "flight.md").write_text("The helicopter needed a new rotor.")

    assert paths(search, "helicopter") == {"flight.md"}


def test_a_note_removed_later_stops_being_found(branch):
    search = KeywordSearch(branch)
    assert paths(search, "corpus") == {"compiler.md"}

    (branch / "compiler.md").unlink()

    assert paths(search, "corpus") == set()


def test_a_note_edited_later_is_found_in_its_new_form(branch):
    search = KeywordSearch(branch)
    assert paths(search, "helicopter") == set()

    (branch / "compiler.md").write_text("Now it is about a helicopter rotor.")

    assert paths(search, "helicopter") == {"compiler.md"}
    assert paths(search, "corpus") == set()


# --- It reports what it left out ---------------------------------------

def test_search_skips_a_file_that_is_not_text(branch):
    """A file that cannot be read as text is not a note to search."""
    (branch / "blob.dat").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    search = KeywordSearch(branch)

    assert paths(search, "corpus") == {"compiler.md"}


def test_search_reports_what_it_could_not_index(branch):
    (branch / "blob.dat").write_bytes(b"\x00\x01\xff\xfe")

    search = KeywordSearch(branch)

    assert [u.path for u in search.unreadable] == ["blob.dat"]


def test_a_branch_of_readable_text_reports_nothing_unreadable(branch):
    assert KeywordSearch(branch).unreadable == []


# --- The edges of it ---------------------------------------------------

def test_search_over_an_empty_branch_finds_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert KeywordSearch(empty).search("anything") == []


def test_search_finds_nothing_in_a_branch_of_directories_only(tmp_path):
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)

    assert KeywordSearch(root).search("anything") == []


def test_search_can_be_asked_for_only_so_many(branch):
    search = KeywordSearch(branch)

    assert len(search.search("bread", limit=1)) == 1


def test_search_survives_a_query_of_punctuation_and_quotes(branch):
    """A reader may type anything, and a syntax error from the baseline
    would look like a finding."""
    search = KeywordSearch(branch)

    assert paths(search, 'the "corpus" (pipeline) [and] *') == {"compiler.md"}


# --- It reads the branch, never the graph -------------------------------

# The words this baseline drops are counted over the branch it is given,
# not read from the index. If they were read from the index, a question
# the graph gets wrong would be a question the baseline also gets wrong,
# and the comparison would be blind to the failure it exists to catch.

def test_the_baseline_needs_no_store_to_search(branch, tmp_path):
    """The comparison is against plain keyword search on the same branch.
    A baseline that needed the graph to exist would not be a baseline."""
    search = KeywordSearch(branch)

    assert paths(search, "corpus") == {"compiler.md"}
    assert not list(tmp_path.glob("*.sqlite3")), "and it wrote none"


def test_the_baseline_counts_the_branch_as_it_stands_and_not_as_the_index_read_it(
    tmp_path,
):
    """The two are counted separately, and can therefore disagree.

    Three notes all about kettles, and the index reads them: "kettle" is in
    every note, so the graph drops it. Then a fourth note arrives about a
    bedroom window. The baseline reads four notes, so "kettle" is in three
    of them and is now the one word that finds them.

    That is the arrangement working. The baseline is an independent
    reading of the same notes rather than a second view of the graph's
    opinion, and it is the only way the two can be found to disagree —
    which is what the comparison is for.
    """
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("a kettle boils")
    (notes / "b.md").write_text("a kettle cools")
    (notes / "c.md").write_text("a kettle stands")
    idx = Index(branch=notes, store_path=tmp_path / "graph.sqlite3")
    idx.refresh()

    (notes / "d.md").write_text("a bedroom window is old")
    baseline = KeywordSearch(notes)

    assert "kettle" in idx.stopwords()
    assert "kettle" not in baseline.stopwords()
    assert paths(baseline, "kettle") == {"a.md", "b.md", "c.md"}


# --- The counting is done once per state of the branch -------------------

def test_the_baseline_counts_its_ubiquitous_words_once_per_branch_and_not_once_per_query(
    branch, monkeypatch
):
    """Which words this branch writes in every note is a fact about the
    notes, so a long list of questions against a branch that has not
    moved re-counts nothing.

    A run over a real corpus is a long list of questions, and each one
    asks the baseline what to search for. Counting per query would
    re-read and re-tokenise every note once per question, which is the
    shape of a cost that grows with the corpus and never gets cheaper
    and never gets noticed.
    """
    counted = []
    counting = acciughe.keyword.stopwords

    def watch(corpus, **kwargs):
        counted.append(len(corpus))
        return counting(corpus, **kwargs)

    monkeypatch.setattr(acciughe.keyword, "stopwords", watch)
    search = KeywordSearch(branch)

    for _ in range(5):
        search.search("kettle bread")

    assert len(counted) == 1, "five questions counted the branch five times"
    assert counted[0] == 12, "and it counted the whole branch, not one note"


def test_the_baseline_counts_again_once_the_branch_has_moved(branch, monkeypatch):
    """The count is cached, not kept: a note that arrives changes which
    words are ubiquitous, and a baseline that kept its answer would go on
    searching a branch that no longer exists."""
    counted = []
    counting = acciughe.keyword.stopwords

    def watch(corpus, **kwargs):
        counted.append(len(corpus))
        return counting(corpus, **kwargs)

    monkeypatch.setattr(acciughe.keyword, "stopwords", watch)
    search = KeywordSearch(branch)
    search.search("kettle")
    assert len(counted) == 1

    (branch / "more.md").write_text("A brass kettle and a birch bed.")
    search.search("kettle")

    assert len(counted) == 2
    assert counted[1] == 13, "it counted the branch as it now stands"
