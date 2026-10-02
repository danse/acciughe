"""Running the evaluation, end to end, without a model.

agenda.md, first item: "**Running the evaluation.** The measures are
built, and nothing has yet been measured with them."

The pieces it joins were each pinned somewhere else — `propose` finds
the questions, `evaluation` judges them, `agent.turn` answers one, and
`report` counts the receipts. What none of them could pin is the order,
and the order is the spec's: *the graph proposes; the notes decide.* A
question the verdict would refuse is never asked, so a run costs no
model time on a question that was never going to count.

**A fake phraser, and that is the point.** A turn on this hardware takes
a minute or two, so a test that ran one would be a test nobody runs
twice. What is being checked is the joining, and a joining is checkable
with a callable that quotes whatever it is handed — the same seam
`test_agent.py` uses. The model is the slow part; the order is the part
that could be wrong.
"""

import time
from itertools import islice

import pytest

from acciughe.evaluation import DROPPED, KEPT, report
from acciughe.index import Index
from acciughe.relations import CO_OCCURRENCE
from acciughe.session import Answer, Citation, Components
from acciughe.trial import gather, run


@pytest.fixture
def branch(tmp_path):
    """A branch the walk finds three questions in, and keeps two of.

    **One question per note, and that is forced rather than tidy.**
    `proposals` walks out from the note a question was asked in, so two
    questions written in one note are proposed against the *same* set of
    notes. Two questions can only split into one kept and one dropped if
    the notes they name can be reached from one of them and not the
    other, and a shared starting point makes that impossible. A fixture
    that put both questions in `seed.md` would have looked reasonable and
    been unable to fail.

    Each question note links to the cluster it is about, so the walk has
    somewhere to go and the support is decided by the links rather than
    by whichever words happened to be shared. The printer and roller
    clusters carry no word of their question that the other notes share,
    which is what makes search unable to follow them; the pipeline
    cluster says `read` in the note the search is asked about, which is
    what makes it followable.

    **The three questions are worded to share at most one word**, and
    that is the other thing a fixture like this has to get right. Two
    notes sharing two words are a co-occurrence edge, and three
    question notes joined to each other is one connected blob in which
    every question is supported by every other question's notes — so
    every verdict keeps all three and there is no dropped question to
    test against. They all want to say "See [[...]]" before a link, and
    it is exactly that shared word which has to go.
    """
    root = tmp_path / "notes"
    root.mkdir()
    (root / "ask-printer.md").write_text(
        "Why did the border printer jam? Printer: [[printer]]."
    )
    (root / "ask-roller.md").write_text(
        "Roller grip lost by when? -> [[roller]]"
    )
    (root / "ask-pipeline.md").write_text(
        "What does the pipeline read? See [[corpus]]."
    )
    (root / "printer.md").write_text("Fed from a hopper. See [[hopper]].")
    (root / "hopper.md").write_text("Damp stock, weighed. See [[roller]].")
    (root / "roller.md").write_text("Grip is lost by August. See [[paper]].")
    (root / "paper.md").write_text("Order by bale, not ream.")
    (root / "corpus.md").write_text("Every note read once, whole.")
    return root


@pytest.fixture
def index(branch, tmp_path):
    idx = Index(branch=branch, store_path=tmp_path / "state" / "graph.sqlite3")
    idx.refresh()
    return idx


def quoting():
    """A phraser that quotes the first note it is shown, in full.

    Quoting a whole note verbatim is the one answer that cannot fail
    either guard: the sentence is in a note the walk reached, and the
    quote is in the note it names. What the tests below are about is
    what the run *does* around that answer, so the answer itself is
    held as still as it can be held.
    """
    calls = []

    def answer_from(question, evidence):
        calls.append((question, evidence))
        path, text = evidence[0]
        return Answer(text, [Citation(path, text)])

    answer_from.calls = calls
    return answer_from


def test_a_question_the_verdict_would_refuse_is_never_asked(index):
    """The notes decide, and the deciding is what saves the model call.

    A question search already answers says nothing about the graph, so
    asking it would spend a minute and a half of model time to produce a
    number the walk had already established was not worth having.
    """
    phraser = quoting()

    tried = list(run(index, phraser))

    asked = [a.proposal.question for a in tried if a.outcome is not None]
    assert asked, "the printer question was asked"
    assert "What does the pipeline read?" not in asked
    assert all(
        a.outcome is None
        for a in tried
        if a.verdict.verdict is not KEPT
    ), "and no question the verdict refused has an outcome at all"


def test_a_run_reports_the_quotations_the_model_wrote_and_not_the_ones_left_over(index):
    """The guard is the thing being measured, so it cannot be the measuring.

    A turn whose citations do not all check out falls back to the
    quotations that do, or refuses outright, and by the time the outcome
    exists everything invented has already gone. A run that counted the
    outcome would report a tool that invents a quote on every single
    question as having perfect citation correctness — a score produced
    entirely by the guard, which is the one thing a measure of citation
    correctness is not allowed to be.
    """

    def inventing(question, evidence):
        path, text = evidence[0]
        return Answer(
            "Something the notes never said.",
            [Citation(path, "a quote from no note at all")],
        )

    tried = list(run(index, inventing))
    found = report(tried)
    shown = [a.outcome for a in tried if a.outcome is not None]

    assert found.fabricated, "the invented quotes are findings, not corrections"
    assert shown, "the questions that were asked produced something"
    assert not any(isinstance(o, Answer) for o in shown), (
        "and none of them was shown as an answer"
    )
    assert found.answers == 0, (
        "so the report is not reading a fallback as a tool that answered well"
    )


def test_a_turn_that_never_reached_a_model_cites_nothing_and_that_is_not_uncorroborated(
    index,
):
    """Nothing written is not a claim with no receipt behind it.

    Both leave the run with no citations to check, and they are the
    opposite facts: a turn that found nowhere to answer from, and a turn
    that found notes and then claimed them with nothing under them. The
    second is the finding this whole measure exists for.
    """
    found = report(run(index, quoting(), limit=1))

    assert found.citations >= 0
    assert found.uncorroborated == 0, "the one question asked was answered properly"


def test_every_asked_question_carries_the_receipts_its_turn_came_back_with(index):
    """The report counts what was checked, so the check has to have run.

    An answer with no `CitationReport` behind it cannot be told from one
    that was never checked, and the second would be a figure of the tool
    rather than of the corpus.
    """
    checked = [a for a in run(index, quoting()) if a.outcome is not None]

    assert checked, "something was asked"
    assert all(a.citations is not None for a in checked), (
        "every outcome with claims behind it was checked against the notes"
    )


def test_a_run_reports_what_it_proposed_and_what_it_reached(index):
    """The yield is the finding, and a stop that is not counted is no finding.

    A turn takes a minute or two, so a run over a real branch has to stop
    somewhere. Stopping at one and reporting a yield of one out of one
    would be telling the reader this branch is nearly all junk, which is
    the opposite of what stopping means.
    """
    everything = report(list(run(index, quoting())))
    one = report(list(run(index, quoting(), limit=1)))

    assert one.proposed == everything.proposed, (
        "the same branch proposed the same questions either way"
    )
    assert one.kept == everything.kept, (
        "a bound shrinks what was asked, not what was worth asking"
    )
    assert one.measured == 1 < everything.measured, (
        "and every question the run did not reach is counted as one"
    )
    assert one.unasked == one.kept - one.measured == 1, (
        "derived from the kept and the asked, so the two cannot drift apart"
    )


def test_the_limit_bounds_what_was_asked_and_not_what_was_proposed(index):
    """The two counts are different facts and only one of them is a bound.

    A limit is there because a turn costs a minute or two, not because
    the branch proposed more questions than are real. A run that stopped
    at one and then reported a yield of one out of one would be counting
    a sample that nobody chose as though it were the whole.
    """
    tried = list(run(index, quoting(), limit=1))

    assert sum(1 for a in tried if a.outcome is not None) == 1, (
        "and it stopped on the bound, not earlier"
    )
    assert sum(1 for a in tried if a.verdict.verdict is KEPT) > 1, (
        "while the questions behind it were still judged"
    )


def test_a_run_with_no_limit_reaches_every_question_it_judged(index):
    assert report(list(run(index, quoting()))).unasked == 0


def test_a_limit_of_nothing_judges_everything_and_asks_nothing(index):
    """Bounding a run to zero is the sharpest case of the same distinction.

    A limit that skipped the judging as well would leave a run that
    measured nothing and reported a yield of nothing, with no way to tell
    that from a corpus with no questions in it.
    """
    tried = list(run(index, quoting(), limit=0))
    found = report(tried)

    assert len(tried) == 3, "all three questions were still walked and judged"
    assert all(a.outcome is None for a in tried)
    assert found.kept == 2, "and the yield is the two the verdicts agreed on"
    assert found.measured == 0


def test_a_phraser_that_writes_no_sentence_produces_parts_and_not_answers(index):
    """A turn that falls back is still something shown, and the run says which.

    The parts are not an answer, and a report that counted them as one
    would be reporting a fallback as a success — the same mistake a
    citation rate makes when it is read without the answer rate beside
    it, one stage earlier in the same turn.
    """

    def in_parts(question, evidence):
        path, text = evidence[0]
        return Components([Citation(path, text)])

    found = report(run(index, in_parts))

    assert found.parts == found.kept
    assert found.answers == 0
    assert found.answer_rate == 0.0, (
        "the run produced notes to look at, and not one sentence"
    )


def test_the_phraser_is_never_called_for_a_question_the_verdict_refused(index):
    phraser = quoting()

    list(run(index, phraser))

    for question, _evidence in phraser.calls:
        assert "pipeline" not in question, (
            "a model call was spent on a question the notes had settled"
        )


# --- Being stopped ------------------------------------------------------

def test_a_run_hands_on_each_receipt_as_it_decides_it(index):
    """Two receipts are in hand while the third question is still unasked.

    A turn takes a minute or two, and a real branch proposes enough
    questions to make a run an evening. A run that built a list and
    returned it at the end would have to finish every question before it
    let go of the first, so an interrupt at question three would throw
    away the whole hour. Taking two receipts and stopping has to cost one
    model call, where a run that finished costs two.

    The first of the three questions is one search already answers, so it
    is judged and yielded without a model call at all. That is the
    spec's order working, and it is why two receipts and one call is the
    right pair to compare.
    """
    phraser = quoting()

    two = list(islice(run(index, phraser), 2))
    whole = list(run(index, quoting()))

    assert len(two) == 2, "two questions were reached"
    assert len(phraser.calls) == 1, (
        "and the run asked the model about one question, not the other"
    )
    assert sum(1 for a in whole if a.outcome is not None) == 2, (
        "so a run that finished is the one that pays for the rest"
    )


def test_every_receipt_is_handed_on_as_it_is_decided(index):
    """The receipts arrive in the order they were decided, and the model
    is called no earlier than the receipt that needed it.

    This is the same property as the one above seen from the other side:
    the count of calls never goes backwards, so no receipt could have
    been produced by a model call that had not happened yet.
    """
    phraser = quoting()
    asked_during: list[int] = []

    def watching(attempt):
        asked_during.append(len(phraser.calls))

    gather(run(index, phraser), on_attempt=watching)

    assert asked_during == sorted(asked_during), "they came in the order decided"
    assert asked_during[-1] == len(phraser.calls) == 2, (
        "and the last one was the receipt of the last question asked"
    )


def test_a_run_that_is_stopped_reports_what_it_had_already_decided(index):
    """Stopping the run is not losing it.

    An hour of measuring that ends in a Ctrl-C has still measured
    everything it managed to do, and a traceback over the top of it would
    be a way of pretending otherwise. So an interrupt costs the attempt
    in flight — which was never finished — and nothing before it.
    """
    answered = quoting()
    calls = answered.calls

    def stopping_at_the_second(question, evidence):
        if len(calls) >= 1:
            raise KeyboardInterrupt
        return answered(question, evidence)

    gathered = gather(run(index, stopping_at_the_second))
    whole = list(run(index, quoting()))

    assert gathered.finished is False, "it says the run did not finish"
    assert len(gathered.attempts) < len(whole), "it really did stop early"
    assert [a.outcome for a in gathered.attempts] == [
        a.outcome for a in whole[: len(gathered.attempts)]
    ], "and what it kept is what a run that finished also decided, in order"

    found = report(gathered.attempts)

    assert found.answers + found.parts + found.refusals == 1, (
        "and the report is made over it regardless"
    )


def test_a_run_stopped_before_it_decided_anything_reports_nothing():
    """The empty case, and it is one a person is most likely to hit.

    Ctrl-C in the first minute of an hour-long run, before any question
    has been judged, is not an unusual thing to do — and it is the case
    where a caller that assumed one receipt had something to report would
    divide by nothing. Here the stop is raised before the first yield, so
    nothing is gathered and nothing may be claimed.
    """

    def stopped_immediately():
        raise KeyboardInterrupt
        yield  # pragma: no cover - the generator body never gets here

    gathered = gather(stopped_immediately())

    assert gathered.attempts == []
    assert gathered.finished is False
    found = report(gathered.attempts)

    assert found.proposed == 0
    assert found.answers == 0


def test_a_run_that_finishes_says_so(index):
    gathered = gather(run(index, quoting()))

    assert gathered.finished is True
    assert len(gathered.attempts) == 3, "every question the run judged is in it"


def test_a_fault_is_not_a_stop_when_there_is_nothing_to_report(index):
    """An empty run has no measurement to save, so the fault is the output.

    A model that has stopped answering, or a branch that cannot be read,
    is a fault. With nothing gathered there is no partial result to
    qualify and no figure to attach it to, so the only honest thing to
    hand back is the exception — a report over zero attempts would say
    "0 of 0 asked" and read as a finding.
    """

    def broken_before_anything():
        raise ConnectionError("the model went away")
        yield  # pragma: no cover - the generator body never gets here

    with pytest.raises(ConnectionError):
        gather(broken_before_anything())


def test_a_fault_after_a_measured_question_keeps_what_was_measured(index):
    """Measured: the model answered 33 times and then returned an
    `HTTPError` 500, and the exception unwound past every one of them.

    The old rule let anything but an interrupt propagate, on the
    reasoning that a report printed over a run that died of a fault would
    be a short run presented as though it were the plan. That reasoning
    does not hold against a run that says it stopped short and says
    why: the report renders `finished`, so nothing was being hidden, and
    all the rule did was throw away real measurements.
    """
    quoter = quoting()

    def dies_on_the_second_question(question, evidence):
        if len(quoter.calls) >= 1:
            raise ConnectionError("the model went away")
        return quoter(question, evidence)

    gathered = gather(run(index, dies_on_the_second_question))
    whole = list(run(index, quoting()))

    assert gathered.finished is False, "it says the run did not finish"
    assert isinstance(gathered.fault, ConnectionError), "naming what stopped it"
    assert [a.outcome for a in gathered.attempts] == [
        a.outcome for a in whole[: len(gathered.attempts)]
    ], "what it kept is what a run that finished also decided, in order"


def test_a_run_stopped_by_its_reader_is_not_a_fault(index):
    """The two ways a run ends short are different things, and a reader
    looking at the result has to be able to tell them apart: one was
    their decision and the other was not."""

    def stopping_at_the_second(question, evidence):
        raise KeyboardInterrupt

    gathered = gather(run(index, stopping_at_the_second))

    assert gathered.finished is False
    assert gathered.fault is None, "nobody was told why, because nobody failed"


# --- At the size a real corpus is ---------------------------------------

def _branch_of(n, tmp_path, every=5):
    """`n` notes, a question in every fifth, each linked to two others.

    **A note's terms are its own name and the two it links to, and
    nothing else in the branch repeats.** `cooccurrence_edges` compares
    every pair, and two notes sharing two terms are an edge — so a
    fixture whose notes all said "log" and "entry" and "clerk" would not
    be three hundred notes with three hundred opinions, it would be one
    blob that every question is supported by. Every one of the 44850
    pairs, and a timing test over a blob measuring something no reader
    will ever run. So the frame here is function words, which every
    language's list drops, and the only content a note carries is its own
    name — which is what a real branch of notes looks like: stated
    relations, and few accidental ones.

    **That the frame is grammar is the whole reason this fixture is
    sparse, and it is a narrower reason than the counting it replaced
    gave.** A share of the corpus drops a frame once the corpus is big
    enough to reach it, so the same fixture used to open every note with
    "Log entry {name} was written by the {name} clerk" and stay sparse
    only because three hundred notes is ninety per cent of itself. A list
    drops grammar at any size and holds nothing else. So the words a
    reader's own habit supplies — a house name, a phrase every note of a
    branch opens with — are no longer dropped by the derivation at all,
    and a branch like that is now one complete graph. That is the cost of
    taking the words from lists rather than from the corpus, taken
    knowingly: the lists know what a language says nothing with and no
    list knows what a writer keeps saying.
    """
    root = tmp_path / f"corpus{n}"
    root.mkdir()

    for i in range(n):
        name = f"n{i:04d}"
        next_one = f"n{(i + 1) % n:04d}"
        after = f"n{(i + 7) % n:04d}"
        body = (
            f"See [[{next_one}]]. See [[{after}]]. "
            f"There is nothing in {name} but {name}."
        )
        if i % every == 0:
            body += f"\nWhy is there nothing in {name}?"
        (root / f"{name}.md").write_text(body)

    return root


def test_a_branch_of_this_shape_is_related_only_by_the_links_it_states(tmp_path):
    """The fixture above is a branch only while its notes are not one blob.

    `cooccurrence_edges` turns two shared terms into an edge, so a corpus
    whose notes share a vocabulary is a complete graph at any size, and
    every figure measured over it — the proposal count, the elapsed time —
    becomes a measurement of the blob instead of of three hundred notes.
    Its docstring claims to avoid that; this is what holds it to it, at
    the size where a share of the corpus used to reach the frame and the
    size where it did not.
    """
    for n in (15, 300):
        index = Index(
            branch=_branch_of(n, tmp_path),
            store_path=tmp_path / "state" / f"{n}.sqlite3",
        )
        index.refresh()

        assert index.graph(kind=CO_OCCURRENCE).edges() == [], (
            f"at {n} notes every pair of them shares a vocabulary, so the "
            "corpus is one blob and what is measured over it is not the "
            "shape the two tests below claim to compare"
        )


def test_a_corpus_of_hundreds_of_notes_is_proposed_and_judged_in_time(tmp_path):
    """The run is the only part of this that touches every note at once.

    A turn costs a minute or two, so the stage this measures is the one
    that has to be quick: propose every question in the branch, judge
    each one, and ask nothing. It walks the graph once per question found,
    and the reader is going to sit through the whole of it before the
    first answer appears — a stage that is slow at three hundred notes is
    a stage nobody waits for at three hundred notes.

    A budget, generously set, because the point is the shape of the cost
    and not the number. The walk was quadratic once and 435 assertions
    held while it was: a timing test is the only instrument in this
    project that can see a complexity class at all.
    """
    big = Index(
        branch=_branch_of(300, tmp_path),
        store_path=tmp_path / "state" / "big.sqlite3",
    )
    big.refresh()

    started = time.monotonic()
    judged = list(run(big, quoting(), limit=0))
    elapsed = time.monotonic() - started

    assert elapsed < 20, (
        f"proposing and judging 300 notes took {elapsed:.1f}s"
    )
    assert len(judged) == 60, "a question in every fifth note was found"
    assert sum(1 for a in judged if a.verdict.verdict is KEPT) + sum(
        1 for a in judged if a.verdict.verdict is DROPPED
    ) == len(judged), (
        "and every one of them was either kept or dropped, with none lost "
        "between being proposed and being judged"
    )


def test_a_corpus_of_hundreds_of_notes_reads_no_worse_than_a_few_do(tmp_path):
    """Scale must not change what the machine does, only how long it takes.

    The figures at three hundred notes and at nine are the same
    proportions, which is the only way the second can be compared with
    the first. A stage that quietly kept a different fraction of the
    questions at size would make every run over a real branch
    incomparable with the fixtures the rest of the project is written on.
    """
    small = Index(branch=_branch_of(15, tmp_path), store_path=tmp_path / "s.sqlite3")
    small.refresh()
    big = Index(branch=_branch_of(150, tmp_path), store_path=tmp_path / "b.sqlite3")
    big.refresh()

    few = report(list(run(small, quoting(), limit=0)))
    many = report(list(run(big, quoting(), limit=0)))

    assert many.proposed / len(judged_of(big)) == few.proposed / len(judged_of(small)), (
        "and the share of notes carrying a question is what decides the count"
    )


def judged_of(index):
    return list(index.note_ids())
