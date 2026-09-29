"""Measuring, rather than assuming.

Encodes the Evaluation section of product.md:

  What matters most is citation correctness: every claim traces to a
  note. Whether an answer is right is not the measure, since judging it
  means already knowing the answer, which is the thing in doubt. The
  set includes questions only search can answer, so the comparison is
  not won by default.

  An agent walking the graph proposes the questions, so they come from
  the corpus rather than from memory. The graph proposes; the notes
  decide: a question counts only when the notes support it, or a
  structure the reader invented becomes its own proof.

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.

No model is consulted anywhere in this module. The semantic half of
"the notes decide" is left unimplemented rather than faked; what is
here is the part that can be checked without one, which is otherwise
the part taken on trust.
"""

import pytest

from acciughe.evaluation import (
    KEPT,
    DROPPED,
    MIN_COMPARISON,
    Attempt,
    CitationReport,
    Finding,
    Verdict,
    citation_report,
    graph_metrics,
    question_verdict,
    report,
)
from acciughe.index import Index
from acciughe.keyword import KeywordSearch
from acciughe.propose import Proposal
from acciughe.session import Answer, Citation, Components, Refusal, RefusalKind


@pytest.fixture
def branch(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "compiler.md").write_text(
        "The compiler pipeline reads a corpus and builds a graph from it."
    )
    (root / "index.md").write_text("Indexing a branch is a stat, not a read.")
    (root / "bread.md").write_text("Butter, flour, water, salt.")
    return root


@pytest.fixture
def index(branch, tmp_path):
    idx = Index(branch=branch, store_path=tmp_path / "state" / "graph.sqlite3")
    idx.refresh()
    return idx


@pytest.fixture
def search(branch):
    return KeywordSearch(branch)


def pair(tmp_path, name, extra=0):
    """``extra`` notes related to nothing, so density can be counted."""
    root = tmp_path / name
    root.mkdir()
    (root / "a.md").write_text("See [[b]].")
    (root / "b.md").write_text("The target of the link.")
    for i in range(extra):
        (root / f"z{i}.md").write_text(f"lonely{i} unique{i} aside{i}")
    idx = Index(branch=root, store_path=tmp_path / f"{name}.sqlite3")
    idx.refresh()
    return idx


@pytest.fixture
def mixed(tmp_path):
    """A branch with a stated link, a shared vocabulary, and an orphan."""
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "a.md").write_text("A note about pipelines and compilers.")
    (root / "b.md").write_text("Another note about pipelines and compilers.")
    (root / "c.md").write_text("See [[a]] for the detail.")
    (root / "d.md").write_text("Nothing in common with anyone here at all.")
    idx = Index(branch=root, store_path=tmp_path / "state" / "mixed.sqlite3")
    idx.refresh()
    return idx


# --- Citation correctness: every claim traces to a note ---------------

def test_a_citation_to_a_real_quote_in_a_real_note_is_correct(index):
    """Every claim traces to a note."""
    answer = Answer(
        text="the pipeline builds a graph",
        evidence=[Citation(path="compiler.md", quote="builds a graph")],
    )

    report = citation_report(answer, index)

    assert report.correct == 1
    assert report.fabricated == []


def test_a_citation_to_a_note_that_does_not_exist_is_incorrect(index):
    answer = Answer(
        text="a claim",
        evidence=[Citation(path="nowhere.md", quote="anything")],
    )

    report = citation_report(answer, index)

    assert report.correct == 0
    assert [f.path for f in report.fabricated] == ["nowhere.md"]


def test_a_citation_whose_quote_is_not_in_the_note_is_fabricated(index):
    """The sharpest check available: the quote has to actually be there.

    A citation naming a real note but quoting text the note does not
    contain is a claim with a plausible-looking receipt.
    """
    answer = Answer(
        text="a claim",
        evidence=[
            Citation(path="compiler.md", quote="uses a quantum annealer")
        ],
    )

    report = citation_report(answer, index)

    assert report.correct == 0
    assert [f.path for f in report.fabricated] == ["compiler.md"]


def test_a_quote_taken_from_another_note_is_fabricated(index):
    """The right text in the wrong note is still invented evidence."""
    answer = Answer(
        text="a claim",
        evidence=[Citation(path="index.md", quote="builds a graph")],
    )

    report = citation_report(answer, index)

    assert report.correct == 0
    assert "index.md" in report.fabricated[0].path


def test_an_answer_citing_nothing_is_entirely_uncorroborated(index):
    """Answers cite their evidence. A claim with no citation has nothing
    behind it."""
    answer = Answer(text="the graph is trustworthy", evidence=[])

    report = citation_report(answer, index)

    assert report.total == 0
    assert report.correct == 0
    assert report.uncorroborated is True


def test_citation_correctness_is_a_fraction_of_what_was_cited(index):
    answer = Answer(
        text="three claims",
        evidence=[
            Citation(path="compiler.md", quote="builds a graph"),
            Citation(path="index.md", quote="stat, not a read"),
            Citation(path="bread.md", quote="uses a quantum annealer"),
        ],
    )

    report = citation_report(answer, index)

    assert (report.total, report.correct) == (3, 2)
    assert report.fraction == pytest.approx(2 / 3)


def test_citation_correctness_ignores_how_good_the_answer_reads(index):
    """Whether an answer is right is not the measure.

    An answer that cites real text is well evidenced whatever it says
    about it; judging the claim would mean already knowing the answer,
    which is the thing in doubt.
    """
    answer = Answer(
        text="butter is the wrong ingredient for a graph",
        evidence=[Citation(path="bread.md", quote="Butter, flour, water, salt")],
    )

    assert citation_report(answer, index).correct == 1


def test_a_citation_is_correct_regardless_of_surrounding_whitespace(index):
    """DECISION: a quote has to appear in the note, not match it byte for
    byte. Whitespace is not a claim, and insisting on byte equality would
    report a true citation as fabricated.

    The leniency stops at whitespace: a quote that differs in case or
    wording is still invented, and nothing here forgives that.
    """
    answer = Answer(
        text="a claim",
        evidence=[Citation(path="bread.md", quote="Butter,   flour")],
    )

    assert citation_report(answer, index).correct == 1


def test_a_report_says_what_it_rejected_and_why(index):
    """An examination says what it left out."""
    answer = Answer(
        text="two claims",
        evidence=[
            Citation(path="compiler.md", quote="builds a graph"),
            Citation(path="nowhere.md", quote="anything"),
        ],
    )

    report = citation_report(answer, index)

    assert report.fabricated[0].path == "nowhere.md"
    assert "does not exist" in report.fabricated[0].reason


# --- The question set, so the comparison is not won by default -------

def test_a_question_search_can_answer_is_dropped(search, index):
    """The set includes questions only search can answer, so the
    comparison is not won by default."""
    verdict = question_verdict(
        text="what does the compiler pipeline build?",
        supported_by=["compiler.md"],
        search=search,
        index=index,
    )

    assert verdict.verdict is DROPPED
    assert "search already finds" in verdict.reason


def test_a_question_search_cannot_answer_is_kept(search, index):
    """The reader's own words, which the baseline cannot follow to the
    notes that answer it."""
    verdict = question_verdict(
        text="what did I figure out about how the notes get wired together?",
        supported_by=["compiler.md", "index.md"],
        search=search,
        index=index,
    )

    assert verdict.verdict is KEPT
    assert verdict.reason


def test_a_question_whose_supporting_note_is_absent_is_dropped(search, index):
    """The graph proposes; the notes decide.

    A question the graph proposed about a note that is not there counts
    for nothing, or a structure the reader invented becomes its own
    proof.
    """
    verdict = question_verdict(
        text="a question about something that was never written",
        supported_by=["imagined.md"],
        search=search,
        index=index,
    )

    assert verdict.verdict is DROPPED
    assert "not in the corpus" in verdict.reason


def test_the_corpus_is_consulted_before_the_search(search, index):
    """A question about a note that does not exist is answered by the
    notes, not by the baseline, so the baseline is never asked."""
    verdict = question_verdict(
        text="what does the compiler pipeline build?",
        supported_by=["imagined.md", "compiler.md"],
        search=search,
        index=index,
    )

    assert verdict.verdict is DROPPED
    assert "not in the corpus" in verdict.reason


def test_a_question_with_nothing_behind_it_is_dropped(search, index):
    verdict = question_verdict(
        text="a question with nothing behind it",
        supported_by=[],
        search=search,
        index=index,
    )

    assert verdict.verdict is DROPPED
    assert "no note" in verdict.reason


def test_search_finding_some_but_not_all_support_is_enough_to_keep(search, index):
    """DECISION: the baseline has to fail the question, not merely
    stumble on it.

    A question is dropped when search surfaces every note that answers
    it, because then the baseline wins it and the question tells us
    nothing. Partial overlap is not a win.
    """
    verdict = question_verdict(
        text="what does the compiler pull in?",
        supported_by=["compiler.md", "index.md"],
        search=search,
        index=index,
    )

    assert search.search("what does the compiler pull in?") == ["compiler.md"]
    assert verdict.verdict is KEPT


def test_a_kept_question_says_which_notes_answer_it(search, index):
    verdict = question_verdict(
        text="what did I figure out about how the notes get wired together?",
        supported_by=["compiler.md", "index.md"],
        search=search,
        index=index,
    )

    assert verdict.supported_by == ["compiler.md", "index.md"]


def test_a_dropped_question_says_which_notes_the_search_found(search, index):
    """An examination says what it left out."""
    verdict = question_verdict(
        text="what does the compiler pipeline build?",
        supported_by=["compiler.md"],
        search=search,
        index=index,
    )

    assert verdict.found_by_search == ["compiler.md"]


# --- The graph is measured rather than assumed ------------------------

def test_a_graph_with_no_edges_has_no_density(index):
    """Three notes, none relating to another."""
    metrics = graph_metrics(index)

    assert metrics.edges == 0
    assert metrics.density == 0.0


def test_a_graph_with_no_notes_reports_nothing_rather_than_dividing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    idx = Index(branch=empty, store_path=tmp_path / "state" / "e.sqlite3")
    idx.refresh()

    metrics = graph_metrics(idx)

    assert metrics.notes == 0
    assert metrics.density == 0.0


def test_density_is_mean_degree(tmp_path):
    """DECISION: density is mean degree, not a share of the pairs that
    could exist.

    Two notes, one relation, and every note related to the other is a
    density of one. A ratio against the possible pairs would report the
    same graph as a perfect graph, and the next note added would halve
    it without a single relation changing.
    """
    assert graph_metrics(pair(tmp_path, "tight")).density == pytest.approx(1.0)


def test_density_falls_when_a_note_joins_that_relations_to_nothing(tmp_path):
    """The other half of the same choice. This change really is the
    corpus becoming less connected — one note in five now reaches
    nothing — so mean degree falling is the honest report. A ratio
    against possible pairs would report the identical drop for a corpus
    that grew while staying connected, which is a different thing."""
    metrics = graph_metrics(pair(tmp_path, "loose", extra=3))

    assert metrics.notes == 5
    assert metrics.density == pytest.approx(2 / 5)


def test_density_counts_a_pair_once_however_many_stages_related_it(mixed):
    """Two stages can relate the same two notes; that is one pair."""
    metrics = graph_metrics(mixed)

    assert metrics.edges == 2
    assert metrics.pairs == 2
    assert metrics.density == pytest.approx(1.0)


def test_the_metrics_say_how_much_is_cooccurrence(mixed):
    """How much of it is co-occurrence rather than relation."""
    metrics = graph_metrics(mixed)

    assert sum(metrics.by_kind.values()) == metrics.edges
    assert metrics.by_kind["link"] == 1
    assert metrics.by_kind["cooccurrence"] == 1


def test_the_metrics_count_notes_connected_to_nothing(mixed):
    """How many notes end up connected to nothing."""
    metrics = graph_metrics(mixed)

    assert metrics.notes == 4
    assert metrics.connected == 3
    assert metrics.unconnected == 1


def test_the_metrics_name_the_notes_they_left_out(mixed):
    """An examination says what it left out: a count tells you there is
    one, and not which."""
    assert graph_metrics(mixed).unconnected_notes == ["d.md"]


def test_the_metrics_account_for_every_note(mixed):
    assert graph_metrics(mixed).connected + graph_metrics(mixed).unconnected == (
        graph_metrics(mixed).notes
    )


def test_a_corpus_of_connected_notes_reports_nothing_unconnected(tmp_path):
    assert graph_metrics(pair(tmp_path, "whole", extra=0)).unconnected == 0


# --- The report: correctness is never reported on its own ---------------

def asked(text, outcome, total=0, good=0):
    """A question the verdict kept, and what came of asking it.

    ``total`` and ``good`` are the citation counts rather than the quotes
    themselves, so a test can state a figure of the shape it is about
    without a corpus standing behind it. What is being checked here is the
    shape: whether the counts are kept apart and reported together, which
    is checkable without a single note existing.
    """
    return Attempt(
        proposal=Proposal(question=text, asked_in="seed.md", answers=["a.md"]),
        verdict=Verdict(KEPT, "search cannot follow this"),
        outcome=outcome,
        citations=(
            None if isinstance(outcome, Refusal) else CitationReport(total, good)
        ),
    )


def dropped(text="why?"):
    """A question the verdict would not measure against, and never asked."""
    return Attempt(
        proposal=Proposal(question=text, asked_in="seed.md", answers=["a.md"]),
        verdict=Verdict(
            DROPPED,
            "search already finds every note that answers it",
            ["a.md"],
            ["a.md"],
        ),
    )


def test_a_question_the_verdict_dropped_cannot_also_have_been_asked():
    """Two opposite states that both end in a Refusal, and are not one thing.

    A question no note answered and a question the tool declined to answer
    look the same in a list. The first says the corpus could not answer it;
    the second says the tool found notes and could not make a sentence of
    them. Averaging them together would make the yield of a corpus look
    like the caution of a tool.
    """
    with pytest.raises(ValueError, match="never asked"):
        Attempt(
            proposal=Proposal(question="why?", asked_in="s.md", answers=["a.md"]),
            verdict=Verdict(DROPPED, "search already finds every note"),
            outcome=Refusal(RefusalKind.THIN),
        )


def test_a_kept_question_with_no_outcome_is_one_the_run_stopped_short_of():
    """The other half of the same distinction, which is not an error.

    A run bounded at ten out of four hundred has to say what it did not
    ask, and a question it judged and did not ask is neither a kept
    question answered nor a question dropped: it is the one number that
    keeps a bounded run from being read as a whole one.
    """
    unreached = Attempt(
        proposal=Proposal(question="why?", asked_in="s.md", answers=["a.md"]),
        verdict=Verdict(KEPT, "search cannot follow this"),
    )

    assert report([unreached]).kept == 1
    assert report([unreached]).unasked == 1
    assert report([unreached]).measured == 0


def test_the_report_says_how_many_proposals_were_worth_measuring_at_all():
    """The yield is the denominator the figures are read against.

    A report of nine answers drawn from nine hundred proposals and a
    report of nine answers drawn from nine are different things, and
    without the first number a reader cannot tell them apart — or cannot
    tell that a corpus full of questions yielded almost nothing worth
    asking, which is the finding.
    """
    found = report([asked("a?", Answer("x"), 2, 2)] * 3 + [dropped()])

    assert found.proposed == 4
    assert found.kept == 3
    assert found.answer_rate == 1.0, "and the rate is of the ones that survived"


def test_a_question_the_verdict_dropped_is_in_the_report_with_its_reason():
    """A question set assembled by discarding things cannot be argued with."""
    found = report([dropped("what about the border printer?")])

    assert len(found.dropped) == 1
    assert found.dropped[0].question == "what about the border printer?"
    assert "search already finds" in found.dropped[0].reason


def test_a_refusal_from_a_turn_is_kept_apart_from_a_question_that_was_dropped():
    """Both are counted, and neither is the other.

    The turn refused a question the verdict kept, so the corpus could
    answer it and the tool would not — which is the refusal rate, and a
    different thing from the yield.
    """
    found = report([asked("a?", Refusal(RefusalKind.THIN))] + [dropped()])

    assert (found.refusals, found.kept) == (1, 1)
    assert len(found.dropped) == 1, "and the dropped one has no outcome at all"


def test_the_rate_of_answers_is_over_the_questions_asked_and_not_the_ones_proposed():
    """A dropped question is not a turn that failed to answer.

    Counting it as one would put a question nobody asked into the
    denominator and make the tool look less willing to answer than it is.
    """
    found = report(
        [asked("a?", Answer("x")), asked("b?", Answer("x")), dropped()]
    )

    assert found.measured == 2
    assert found.answer_rate == 1.0, "both asked questions were answered"


def test_an_answer_and_a_set_of_parts_are_counted_apart():
    """A run of fallbacks must not read as a run of answers.

    The two are both something shown to the reader, and they are separate
    outcomes precisely because a sentence the notes do not contain is not
    an answer. Averaging them together would take the distinction back out
    at the one place a reader sees the total.
    """
    found = report(
        [asked("a?", Components([Citation("a.md", "q")]), 1, 1)] * 3
        + [asked("b?", Answer("x"), 1, 1)]
    )

    assert found.answers == 1
    assert found.parts == 3
    assert found.answer_rate == 0.25, (
        "the parts are shown and none of them is a sentence"
    )


def test_an_answer_with_nothing_behind_it_is_counted_where_its_citations_are_not():
    """No citations is the worst result, and it is not a fraction of nothing.

    Averaged in with answers that did cite, an answer that cited nothing
    becomes invisible: the denominator counted the other answers'
    citations, and a claim with no receipt behind it looks like a claim
    whose receipts happened to be right.
    """
    found = report(
        [asked("a?", Answer("x"), 0, 0), asked("b?", Answer("y"), 2, 2)]
    )

    assert found.uncorroborated == 1
    assert found.citations == 2, "the uncorroborated one contributes no citations"


def test_a_refusal_is_not_an_answer_with_nothing_behind_it():
    """A refusal claimed nothing, so there is nothing missing from it.

    Both count as zero citations and the two are not the same fact: one
    is a tool that said it did not know, the other is a tool that spoke
    and could back none of it up.
    """
    found = report([asked("a?", Refusal(RefusalKind.THIN))])

    assert found.citations == 0
    assert found.uncorroborated == 0, "it claimed nothing, so nothing is unsupported"


def test_a_refusal_cannot_be_made_to_raise_the_rate_of_correct_citations():
    """The reason the rate is reported next to the rate of answers at all.

    Refusing everything produces no wrong citations, so a measure of
    citation correctness alone is best served by a tool that never answers
    — a tool that is correct and useless, and whose number looks like a
    success. Here nothing is cited at all, and the run has to be legible
    as a run that declined rather than as a perfect score.
    """
    found = report([asked("a?", Refusal(RefusalKind.THIN))] * 9)

    assert found.answer_rate == 0.0
    assert found.measured == 9
    assert found.citation_rate == 0.0, (
        "and zero of zero is an absence of a score, which is how it is printed"
    )


def test_every_wrong_citation_in_the_run_is_named_and_not_only_the_first():
    """One bad quote in a note leaves that note's other quotes standing,
    so dropping the note would lose them."""
    wrong = Finding("a.md", "quoted text is not in the cited note", "invented")
    first = asked("a?", Answer("x"), 2, 1)
    second = asked("b?", Answer("y"), 2, 1)

    found = report([
        Attempt(first.proposal, first.verdict, first.outcome,
                CitationReport(2, 1, [wrong])),
        Attempt(second.proposal, second.verdict, second.outcome,
                CitationReport(2, 1, [Finding("b.md", "also invented")])),
    ])

    assert [f.path for f in found.fabricated] == ["a.md", "b.md"]


def test_a_set_below_the_floor_is_not_a_comparison():
    """Two questions is not a comparison, and the report has to be unable
    to present one as though it were.

    Not by withholding: a reader who was not shown what happened cannot
    tell silence from a result. By saying so, before the figures, so the
    figures are read as what they are.
    """
    assert report([asked("a?", Answer("x"))] * (MIN_COMPARISON - 1)).is_a_comparison is False
    assert report([asked("a?", Answer("x"))] * MIN_COMPARISON).is_a_comparison is True


def test_the_report_needs_no_corpus_and_asks_no_model():
    """No model is consulted anywhere in this module, and a measure that
    guesses is worse than no measure, since it is believed.

    Which is only worth saying if it is true of the report too, and not
    just of the parts that were written first. Every test above builds
    its figures out of counts and gets a report, with no branch, no store
    and no model: a report that had to go and look would be measuring and
    aggregating in one pass, and there would be nothing in it that could
    guess.
    """
    found = report([asked("a?", Answer("x"), 2, 2)])

    assert (found.answers, found.citations, found.correct) == (1, 2, 2)
