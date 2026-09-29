"""A turn as a reader sees it.

The gap this file closes. `test_agent.py` pins that a turn decides one of
three outcomes and `test_parts.py` pins what the third one contains, but
until now nothing in the project could be *run*: the outcomes were types
with no rendering, so the shape a reader meets was undefined and
therefore untested.

Rendering is a pure function of a `Turn`, which is the only reason these
tests are cheap. A turn on this hardware is one to two minutes, and a
test that has to wait for one is a test that stops being run.

product.md:

  Answers cite their evidence. Every claim points at the notes behind
  it. When a graph does not support an answer, the command says so
  rather than inventing one.

  When the graph does not support an answer, the command says so, and
  says which kind of not-knowing it is: absent from the corpus, present
  but unconnected, or thin. It never dresses a partial result as an
  answer, and never answers with hedging where a clean refusal would
  serve. A refusal keeps the nearest notes, marked as not an answer, and
  is inspectable like any other.

  Within it, a refusal is a fork rather than a wall: what is absent
  ends the search, what is present but unconnected redirects the
  question, what is thin asks for more.

  Inspectable. The structure behind an answer can be examined, and an
  examination says what it left out.

  Both happen unasked, and are reported rather than silent.

The property most of this file is after is that the three outcomes are
not mistakable for one another. A reader who cannot tell a conclusion
from a set of notes has been shown the thing the spec forbids, and the
burden falls on the rendering rather than on the reader being careful.
"""

import pytest

from acciughe.agent import Turn, turn
from acciughe.corpus import Unreadable
from acciughe.evaluation import (
    DROPPED,
    KEPT,
    MIN_COMPARISON,
    Attempt,
    CitationReport,
    Finding,
    Verdict,
    report,
)
from acciughe.index import Index, Read
from acciughe.propose import Proposal
from acciughe.render import render, render_attempt, render_report
from acciughe.session import (
    Answer,
    Citation,
    Components,
    NearNote,
    Refusal,
    RefusalKind,
)

ASKING = "what does the compiler pipeline read?"


@pytest.fixture
def branch(tmp_path):
    """A seed the question lands on and two notes the walk reaches.

    The two are worded differently on purpose: identical bodies would
    make a mis-attributed quote indistinguishable from a correct one, and
    these tests are partly about telling them apart.

    Four notes is too few for a word count to settle, so the words the
    question shares with `lonely` — "what", "is", "the" — are not yet
    ubiquitous and the question about ducks matches it. These tests are
    about how an outcome is rendered, and the outcome under test has to be
    the one they mean; `tests/test_agent.py` keeps the same three notes
    at a size where a count can decide, and asserts the refusal kinds
    there.
    """
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "seed.md").write_text(
        "compiler pipeline corpus. See [[one]] and [[two]]."
    )
    (notes / "one.md").write_text("branch notes about unrelated gardening")
    (notes / "two.md").write_text("branch notes about allotment planning")
    (notes / "lonely.md").write_text("the allotment shed needs a new roof")
    return notes


@pytest.fixture
def unanswered(branch_of_real_notes):
    """A branch where a question about ducks really does match nothing.

    Rendered refusals are worth testing against the refusal they claim to
    be. A four-note branch cannot produce one: its common words are in two
    notes of four and so are not yet shared by the corpus, which makes a
    question about escape velocity land on the allotment shed.
    """
    return branch_of_real_notes


@pytest.fixture
def branch_of_real_notes(tmp_path):
    """Ten notes written the way a branch is written.

    Every note opens with "the" and carries "about", because those are
    the words a note-taking corpus is full of and a branch without them is
    a fixture. The subjects are in two notes each.
    """
    notes = tmp_path / "real"
    notes.mkdir()
    for name, text in [
        ("printer", "the printer jams on duplex about the cartridge."),
        ("printer2", "the printer wants toner about the drum."),
        ("allotment", "the allotment butt needs water about now."),
        ("allotment2", "the allotment fence wants a coat about now."),
        ("ferry", "the ferry timetable changed about the crossing."),
        ("ferry2", "the ferry crossing runs about hourly."),
        ("hall", "the hall ceiling about the winter wants painting."),
        ("shed", "the shed roof about the felt leaked again."),
        ("bus", "the bus is late about the roadworks outside."),
        ("piano", "the piano wants tuning about twice a year."),
    ]:
        (notes / f"{name}.md").write_text(text)
    idx = Index(branch=notes, store_path=tmp_path / "real.sqlite3")
    idx.refresh()
    return idx


@pytest.fixture
def index(branch):
    idx = Index(branch=branch, store_path=branch.parent / "graph.sqlite3")
    idx.refresh()
    return idx


def parts(*pairs):
    def phraser(question, evidence):
        return Components(evidence=[Citation(path=p, quote=q) for p, q in pairs])

    return phraser


def answering(*pairs):
    def phraser(question, evidence):
        return Answer(
            text="The pipeline reads the notes folder.",
            evidence=[Citation(path=p, quote=q) for p, q in pairs],
        )

    return phraser


# --- The three outcomes do not look alike -------------------------------

def test_an_answer_is_a_sentence_with_its_notes_under_it(index):
    result = turn(
        ASKING, index,
        answer_from=answering(("one.md", "about unrelated gardening")),
    )

    shown = render(result)

    assert "The pipeline reads the notes folder." in shown
    assert 'one.md: "about unrelated gardening"' in shown


def test_the_parts_are_shown_with_the_reason_they_are_not_an_answer(index):
    """product.md: "It never dresses a partial result as an answer."

    Both halves are load-bearing. Without the first, the missing sentence
    reads as the tool having found nothing; without the second, what is on
    the screen looks like the result of a successful search.
    """
    result = turn(
        ASKING, index,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    shown = render(result)

    assert "your notes do not contain" in shown
    assert "These are notes, not an answer." in shown


def test_a_refusal_says_which_kind_of_not_awareness_it_is(index):
    """product.md: "says which kind of not-knowing it is: absent from the
    corpus, present but unconnected, or thin."""
    result = turn("what is in the shed?", index, answer_from=parts())

    shown = render(result)

    assert result.outcome.kind is RefusalKind.UNCONNECTED
    assert "not connected to anything you asked about" in shown


def test_the_three_outcomes_share_no_claim_line(index):
    """The load-bearing test of the file.

    Rendered three ways, the parts of the output that assert something
    about the corpus must not overlap. If a reader could read an answer
    as a refusal or the reverse, the distinction the spec spends a
    paragraph on has been left to the wording of three sentences.
    """
    answered = render(turn(
        ASKING, index,
        answer_from=answering(("one.md", "about unrelated gardening")),
    ))
    shown_parts = render(turn(
        ASKING, index, answer_from=parts(("one.md", "about unrelated gardening")),
    ))
    refused = render(turn("what is in the shed?", index, answer_from=parts()))

    assert "The pipeline reads the notes folder." in answered
    assert "The pipeline reads the notes folder." not in shown_parts
    assert "The pipeline reads the notes folder." not in refused

    assert "These are notes, not an answer." not in answered
    assert "These are notes, not an answer." not in refused


def test_a_refusal_never_shows_a_note_the_turn_reached(index):
    """A path in a refusal means "this is the note I could not use". A path
    in an answer means "this is what I used". The same word for both is
    the ambiguity; the fork is what separates them."""
    result = turn("what is in the shed?", index, answer_from=parts())

    assert "lonely.md" in render(result)
    assert 'lonely.md: "' not in render(result)


# --- Citations survive to the screen ------------------------------------

def test_a_multi_line_quote_is_printed_on_one_line(index):
    """A note cannot fake a boundary.

    Raw, a quote containing a newline puts the next note's path inside
    the previous note's quotation, and a reader checking a citation by eye
    is reading a boundary the tool invented.
    """
    (index.branch / "one.md").write_text(
        "branch notes about unrelated gardening\nand about allotment planning"
    )
    long_quote = "about unrelated gardening\nand about allotment planning"

    result = turn(
        ASKING, index, answer_from=answering(("one.md", long_quote)),
    )

    line = next(
        line for line in render(result).splitlines()
        if line.startswith("  one.md")
    )
    assert line.count('"') == 2
    assert "unrelated gardening and about" in line


def test_no_citation_is_dropped_between_the_outcome_and_the_screen(index):
    """product.md: "Answers cite their evidence."

    Every citation that survived the guard is printed, and nothing is
    printed that the outcome did not carry. A renderer that showed the
    first citation, or reformatted a quote into a different string, would
    leave the reader checking something other than what was verified —
    and the guard's number would describe an answer nobody ever saw.
    """
    result = turn(
        ASKING, index,
        answer_from=answering(
            ("one.md", "about unrelated gardening"),
            ("two.md", "about allotment planning"),
        ),
    )

    shown = render(result)
    for citation in result.outcome.evidence:
        assert f'  {citation.path}: "{citation.quote}"' in shown
    assert shown.count(": \"") == 2


# --- The fork -----------------------------------------------------------

def test_a_refusal_offers_the_way_through_it(index):
    """product.md: "a refusal is a fork rather than a wall: what is
    present but unconnected redirects the question, what is thin asks for
    more." A refusal printed without its fork is the wall."""
    result = turn("what is in the shed?", index, answer_from=parts())

    assert "try asking: What is in lonely.md?" in render(result)


def test_what_is_absent_ends_the_search_and_offers_nothing(unanswered):
    """The one refusal that is a wall. Printing a fork here would invite
    the reader to try a question that cannot succeed."""
    result = turn(
        "what is the escape velocity of a duck?", unanswered, answer_from=parts(),
    )

    assert result.outcome.kind is RefusalKind.ABSENT
    assert "try asking" not in render(result)
    assert "this would help" not in render(result)


def test_a_thin_walk_asks_for_the_notes_that_would_have_filled_it(index):
    result = turn(ASKING, index, answer_from=answering())

    assert "this would help" in render(result)


# --- A refusal keeps the nearest notes ----------------------------------

def test_a_refusal_names_the_notes_it_kept_and_why(index):
    """product.md: "A refusal keeps the nearest notes, marked as not an
    answer, and is inspectable like any other."

    The reason is shared by every note in one turn, which is itself the
    check: a reader told "unconnected" can confirm it by opening one.
    """
    result = turn("what is in the shed?", index, answer_from=parts())

    shown = render(result)
    assert "lonely.md" in shown
    assert "reaches nothing" in shown


# --- The walk is shown, including what it left out ----------------------

def test_the_walk_is_shown_with_the_notes_the_model_was_given(index):
    result = turn(
        ASKING, index, answer_from=answering(("one.md", "about gardening")),
    )

    shown = render(result)
    assert "walk:" in shown
    assert "seed.md" in shown
    assert "one.md" in shown


def test_the_walk_says_which_notes_it_left_out(index):
    """product.md: "an examination says what it left out." The bound is
    below the number of notes here, so the walk withheld nothing and the
    line must not claim otherwise."""
    result = turn(
        ASKING, index, answer_from=answering(("one.md", "about gardening")),
    )

    assert "left out" not in render(result)


def test_a_question_landing_nowhere_says_so(unanswered):
    """"no seeds" and "no notes beyond the seeds" are two different
    not-knowings, and an empty list would render as both at once. The
    first is a wall; the second is a walk that got somewhere and found
    nothing to join it to."""
    result = turn(
        "what is the escape velocity of a duck?", unanswered, answer_from=parts(),
    )

    shown = render(result)
    assert result.outcome.kind is RefusalKind.ABSENT
    assert "no note in your corpus uses the words of the question" in shown
    assert "→" not in shown


def test_a_walk_that_reached_only_its_seed_says_what_it_reached(index):
    """The other half of the pair above. One note is not an empty list,
    and printing the seed twice with an arrow between them would dress a
    walk that went nowhere as a walk that went somewhere. Reaching
    exactly where it started *is* the unconnected case."""
    result = turn("what is in the shed?", index, answer_from=parts())

    shown = render(result)
    assert result.outcome.kind is RefusalKind.UNCONNECTED
    assert "walk: lonely.md, reaching nothing beyond it" in shown
    assert "lonely.md → lonely.md" not in shown


# --- The read is reported -----------------------------------------------

def test_a_quiet_check_still_says_the_notes_are_current():
    """product.md: "Both happen unasked, and are reported rather than
    silent." Both is the check and the refresh, so a turn that found
    nothing to do says that too — six words against having to assume the
    graph matches the branch."""
    shown = render(Turn(question=ASKING, outcome=Refusal(RefusalKind.ABSENT)))

    assert "up to date" in shown


def test_a_refresh_says_what_it_did():
    shown = render(
        Turn(
            question=ASKING,
            outcome=Refusal(RefusalKind.ABSENT),
            refresh=Read(read=["a.md", "b.md"], added=["b.md"], rebuilt=True),
        )
    )

    assert "read 2 notes" in shown
    assert "1 new" in shown
    assert "graph rebuilt" in shown


def test_a_file_that_is_not_a_note_is_reported_with_its_reason():
    """The reason matters as much as the path: "not text" and "not here"
    call for different things from the reader, and reporting only the
    path leaves them guessing which one happened."""
    shown = render(
        Turn(
            question=ASKING,
            outcome=Refusal(RefusalKind.ABSENT),
            refresh=Read(
                unreadable=[Unreadable("photo.png", "not text: not utf-8")]
            ),
        )
    )

    assert "photo.png" in shown
    assert "not utf-8" in shown


def test_a_link_to_a_note_that_is_not_there_is_reported():
    shown = render(
        Turn(
            question=ASKING,
            outcome=Refusal(RefusalKind.ABSENT),
            refresh=Read(dangling=[("seed.md", "missing")]),
        )
    )

    assert "missing" in shown


# --- Saying what is being read, while it is being read -----------------

def test_the_reader_is_told_what_is_being_read_before_the_pause(index):
    """A turn is one to two minutes on this hardware and the pause is
    opaque, so a silent terminal and a hung one look identical from the
    outside. Saying the notes before the call is what tells them the
    tool is still working, and where."""
    said = []

    turn(
        ASKING, index,
        answer_from=answering(("one.md", "about unrelated gardening")),
        on_progress=said.append,
    )

    assert "reading 4 notes: lonely.md, seed.md, one.md, two.md" in said[0]
    assert "one to two minutes" in said[0]


def test_a_refused_turn_reports_no_reading(index):
    """Nothing was read by the model, because the model was never called.
    Reporting a read that did not happen would put notes on screen that
    no sentence was ever drawn from."""
    said = []

    turn("what is in the shed?", index, answer_from=parts(),
         on_progress=said.append)

    assert said == []


def test_it_is_said_at_most_once(index):
    """The same rule the model call obeys. A progress sink that could
    fire repeatedly would be something to reason about, and nothing else
    in a turn is worth that."""
    said = []

    turn(
        ASKING, index,
        answer_from=answering(("one.md", "about unrelated gardening")),
        on_progress=said.append,
    )

    assert len(said) == 1


def test_a_turn_needs_no_progress_sink(index):
    """The sink is for a reader waiting on a terminal. A caller that is
    not should not have to pass one, and a `turn` that demanded one would
    be unable to answer without a screen."""
    result = turn(
        ASKING, index,
        answer_from=answering(("one.md", "about unrelated gardening")),
    )

    assert isinstance(result.outcome, Answer)


# --- One note is not two ------------------------------------------------

def test_one_note_is_not_reported_as_two():
    shown = render(
        Turn(
            question=ASKING,
            outcome=Refusal(RefusalKind.ABSENT),
            refresh=Read(read=["only.md"]),
        )
    )

    assert "read 1 note " in shown or "read 1 note\n" in shown
    assert "read 1 notes" not in shown


# --- A report cannot be read as a better result than the run was --------

def _answer_attempt(question, total=2, good=2):
    """One kept question, answered, with the receipts it came back with."""
    return Attempt(
        proposal=Proposal(question=question, asked_in="s.md", answers=["a.md"]),
        verdict=Verdict(KEPT, "search cannot follow this"),
        outcome=Answer("x"),
        citations=CitationReport(total, good),
    )


def run_of(*outcomes, total=2, good=2):
    """A report over a run of turns, each one given the same receipts."""
    return report([
        Attempt(
            proposal=Proposal(question=f"q{n}?", asked_in="s.md", answers=["a.md"]),
            verdict=Verdict(KEPT, "search cannot follow this"),
            outcome=outcome,
            citations=(
                None
                if isinstance(outcome, Refusal)
                else CitationReport(total, good)
            ),
        )
        for n, outcome in enumerate(outcomes)
    ])


def test_the_rate_of_answers_is_on_the_same_line_as_the_figures_it_qualifies():
    """Citation correctness alone is maximised by refusing every question.

    A tool that never answers has no wrong citations and would report a
    perfect score, so the rate at which answers are produced is not a
    second finding beside the correctness — it is what the correctness is
    read over. Printing them apart is what lets one of them be quoted on
    its own, so they are printed together.
    """
    shown = render_report(run_of(Answer("x"), Answer("y")))

    answering = [l for l in shown.splitlines() if l.startswith("  answers")]
    assert len(answering) == 1
    assert "2 of 2 asked" in answering[0], "the rate names its own denominator"
    assert "100%" in answering[0]


def test_a_run_that_answered_nothing_does_not_print_a_score_for_having_been_right():
    """Nine refusals is a tool that declined, and declining is not 0%.

    Nothing was cited, so there is nothing to check. Printing 0% would
    report a tool that correctly said it did not know as though it had
    spoken and been wrong — the opposite failure, and the same mistake of
    reading an absence as a measurement.
    """
    shown = render_report(run_of(*[Refusal(RefusalKind.THIN)] * 9))

    assert "answers 0 of 9 asked (0%)" in shown, "the rate is reported, and is zero"
    assert "nothing was cited, so there is nothing to check" in shown
    assert "in the note they name" not in shown, (
        "a run that made no claim prints no figure for its correctness"
    )


def test_a_run_too_small_to_compare_says_so_before_it_is_read():
    """The reader has to know what they are looking at before they read it.

    A caveat under the figures is a postscript, and a postscript is read
    after the number has already been taken away.
    """
    shown = render_report(run_of(Answer("x"))).splitlines()

    assert "is not a comparison" in shown[1], "the second line, before any figure"
    assert any("100%" in l for l in shown[2:]), "and the figures are still there"


def test_a_set_too_small_to_compare_is_not_withheld():
    """Silence is not a result, and a reader not shown what happened cannot
    tell the two apart."""
    shown = render_report(run_of(Answer("x")))

    assert "1 of 1 proposed question kept" in shown
    assert "answers 1 of 1" in shown


def test_a_figure_is_only_reported_once_there_are_enough_questions_to_report_it():
    assert "is not a comparison" not in render_report(
        run_of(*[Answer("x")] * MIN_COMPARISON)
    )


def test_an_answer_with_nothing_behind_it_is_shown_even_when_every_other_citation_held():
    """It is the worst result available, and averaging it away is the one
    way to make an answer that invents its way past the check respectable.

    The run below answers three questions, two of them with citations that
    all checked out, so the fraction on its own is 100%. The third claims
    with nothing behind it, and it has to be visible against that.
    """
    found = report([
        _answer_attempt("q0", total=0, good=0),
        _answer_attempt("q1", total=2, good=2),
        _answer_attempt("q2", total=2, good=2),
    ])

    shown = render_report(found)

    assert "citations 4 of 4" in shown, "the fraction is a perfect score"
    assert "1 answered with nothing behind them" in shown, (
        "and the claim with no receipt behind it is not inside that score"
    )


def test_a_run_that_stopped_short_of_a_question_says_so_where_the_yield_is():
    """The yield and the stop are one sentence, because they are one fact.

    "Ten of four hundred kept" and "ten of ten kept" are the same yield
    printed over wildly different corpora, and only the second number
    says which one happened. A reader shown the first and not the second
    is being told the branch is nearly all junk, which is what stopping
    at ten of four hundred emphatically is not.
    """
    shown = render_report(
        report(
            [
                Attempt(
                    proposal=Proposal(
                        question=f"q{n}?", asked_in="s.md", answers=["a.md"]
                    ),
                    verdict=Verdict(KEPT, "search cannot follow this"),
                    outcome=Answer("x") if n < 2 else None,
                    citations=CitationReport(2, 2) if n < 2 else None,
                )
                for n in range(42)
            ]
        )
    )

    assert "2 of 42 proposed questions kept, 40 judged and not reached" in shown


def test_a_run_that_reached_everything_does_not_mention_what_it_did_not_reach():
    found = report(
        [
            Attempt(
                proposal=Proposal(question="q?", asked_in="s.md", answers=["a.md"]),
                verdict=Verdict(KEPT, "search cannot follow this"),
                outcome=Answer("x"),
                citations=CitationReport(2, 2),
            )
        ]
    )

    assert "not reached" not in render_report(found)


def test_a_wrong_citation_is_named_by_note_and_why():
    """A report that cannot say which of a note's citations was wrong cannot
    be acted on: one bad quote in a note leaves its other quotes standing."""
    found = report([
        Attempt(
            proposal=Proposal(question="q?", asked_in="s.md", answers=["a.md"]),
            verdict=Verdict(KEPT, "search cannot follow this"),
            outcome=Answer("x"),
            citations=CitationReport(
                2,
                1,
                [Finding("border.md", "quoted text is not in the cited note")],
            ),
        )
    ])

    shown = render_report(found)

    assert "border.md" in shown
    assert "quoted text is not in the cited note" in shown


# --- What a person sees while the run goes on ----------------------------

def _attempt(outcome, citations=None, verdict=KEPT, reason="search cannot follow"):
    return Attempt(
        proposal=Proposal(question="q?", asked_in="press.md", answers=["a.md"]),
        verdict=Verdict(verdict, reason),
        outcome=outcome,
        citations=citations,
    )


def test_a_line_of_progress_is_one_line(index):
    """A run watched for an hour is skimmed, not read.

    Anything longer than a line is not skimmed at all, and a progress
    line that ran to two is a progress line that stops being progress
    when there is a lot of it.
    """
    shown = render_attempt(_answer_attempt("why did the press jam?", 2, 2))

    assert shown.count("\n") == 0
    assert len(shown) < 90, "and it is short enough to skim"


def test_the_line_of_progress_says_whether_the_model_is_still_inventing():
    """The one figure worth watching as the run goes.

    This is the finding the whole evaluation exists to produce, and a
    reader who has to wait an hour to learn that the first ten questions
    all fabricated has no way to stop the run. A model that invents its
    quotes every time is a run to abandon, and the line is the only
    place that shows it while there is still something to save.
    """
    invented = _attempt(
        Answer("x"),
        CitationReport(2, 1, [Finding("tray.md", "not in the note")]),
    )

    shown = render_attempt(invented)

    assert "1 of 2 quotes invented" in shown
    assert "tray.md" not in shown, (
        "the run's own report names the note; the line only carries the count"
    )


def test_a_question_the_model_answered_cleanly_says_so():
    shown = render_attempt(_answer_attempt("why did the press jam?", 2, 2))

    assert "answered" in shown
    assert "2 of 2 quotes in the notes" in shown
    assert "invented" not in shown


def test_the_three_outcomes_are_told_apart_on_the_line_too():
    """The distinction the spec forbids blurring is not blurred by being terse.

    An answer, a set of notes, and a refusal are three different results,
    and a one-word summary that called them all "result" would be a
    reader watching a run unable to tell which of them the tool has been
    producing all evening.
    """
    answered = render_attempt(_attempt(Answer("x"), CitationReport(2, 2)))
    parts = render_attempt(_attempt(Components([Citation("a.md", "q")]),
                                   CitationReport(1, 1)))
    refused = render_attempt(
        _attempt(Refusal(RefusalKind.THIN), None)
    )

    assert "answered" in answered
    assert "parts" in parts and "not an answer" in parts
    assert "refused" in refused and "thin" in refused
    assert len({answered.split(":")[1], parts.split(":")[1], refused.split(":")[1]}) == 3


def test_a_dropped_question_says_why_it_was_never_asked():
    """The yield is a fact about the corpus, and a reader watching the run
    cannot see ninety judged questions without the ones that were not
    worth measuring.

    Nothing was spent on this question, so the line is the verdict and
    nothing else — and in particular it is not phrased as though the run
    had tried and failed.
    """
    shown = render_attempt(
        _attempt(
            None,
            None,
            verdict=DROPPED,
            reason="search already finds the answer",
        )
    )

    assert shown == "press.md: search already finds the answer"
    assert "not asked" not in shown, "it was dropped, not left unreached"


def test_a_question_the_run_never_reached_says_so(index):
    """Different from dropped, and saying so is the whole point.

    A limit stops a run. The question was worth measuring and was not
    measured, and a line that read like a dropped one would make a
    bounded run look like a corpus of junk.
    """
    shown = render_attempt(_attempt(None, None))

    assert shown == "press.md: kept, not asked"


def test_a_question_the_model_never_claimed_anything_from_says_so():
    """Cited nothing is not the same as having been refused.

    A turn that found nowhere to answer from has no claims, and a turn
    that found notes and claimed them with nothing under them has claims
    that were all invented. The second is the finding; the line must not
    let the first hide it.
    """
    shown = render_attempt(_attempt(Components([Citation("a.md", "q")]), None))

    assert "cited nothing" in shown


def test_the_line_names_the_note_the_question_was_asked_in():
    """A run over hundreds of notes produces hundreds of these, and a line
    that does not say which note it came from is a line that cannot be
    looked up afterwards."""
    assert render_attempt(_attempt(Answer("x"), CitationReport(1, 1))).startswith(
        "press.md: "
    )
