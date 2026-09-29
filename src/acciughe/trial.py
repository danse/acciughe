"""Running the evaluation over a branch, question by question.

agenda.md, first item: "**Running the evaluation.** The measures are
built, and nothing has yet been measured with them."

This is the part that makes the measures runnable — the walk, the
verdict, a turn, and a receipt for each, handed on as the list that
``evaluation.report`` counts.

**Its own module, because it is the only one that depends on both the
evaluator and the agent.** ``agent`` imports ``CitationReport`` from
``evaluation``, so a runner inside ``evaluation`` would need to import
``agent`` back, and the two would be one another's import. Here the
dependency points one way, which is also the direction the work goes:
propose, then judge, then ask, then count.

**Every question is judged, and only some are asked.** A turn costs a
minute or two, so a run over a real branch has to stop somewhere, and a
stop that skips the judging would report a yield of ten out of ten when
ten were all that were looked at rather than all that were there. A
question the run never reached is kept in the list with no outcome, which
is what lets the report count it as one rather than as an absence.

Nothing here asks a model what it thinks. The one call made per asked
question is the answering path, which is the thing under test, and its
output is checked against the notes before it is counted.

**A run is stopped by the person running it, and stopping it is not
losing it.** A turn takes a minute or two and a real branch proposes
enough questions to make the run an evening, so it will be interrupted.
So the receipts are handed on as they are decided rather than collected
at the end, and ``gather`` is what takes them: an interrupt costs the
attempt in flight and nothing before it.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from acciughe.agent import Phraser, turn
from acciughe.evaluation import KEPT, Attempt, question_verdict
from acciughe.index import Index
from acciughe.keyword import KeywordSearch
from acciughe.propose import proposals
from acciughe.session import Session


def run(
    index: Index,
    answer_from: Phraser,
    *,
    search: KeywordSearch | None = None,
    session: Session | None = None,
    limit: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[Attempt]:
    """Propose, judge, ask, and hand on each receipt as it is decided.

    The order is the spec's: the graph proposes, the notes decide, and
    only then is anything asked. A question the verdict would not measure
    against is recorded and not asked, so a run costs no model time on a
    question that was never going to count.

    ``limit`` bounds the questions *asked*, not the ones proposed or
    judged, and the difference is reported rather than merely honoured:
    ``evaluation.report`` counts a kept question with no outcome as one
    the run stopped short of, so bounding a run cannot quietly shrink
    the corpus it claims to have measured.

    ``on_progress`` is passed to the turn, which calls it at most once
    per question, immediately before the model call.

    **It is a generator, and that is the reason a person who stops it has
    something.** A turn takes a minute or two and a real branch proposes
    enough questions to run for an evening, so a run will be interrupted
    — and a run that accumulated its receipts into a list would lose
    every one of them to the interrupt, having spent the hour. Yielding
    each one as it is decided hands the work to the caller, which can
    report what it gathered from what has already been gathered. The
    attempt in flight when the interrupt lands is the only one lost, and
    it was never finished anyway.
    """
    looked_at = search or KeywordSearch(index.branch)
    asked = 0

    for proposal in proposals(index):
        verdict = question_verdict(
            proposal.question, proposal.answers, looked_at, index
        )
        if verdict.verdict is not KEPT or (limit is not None and asked >= limit):
            yield Attempt(proposal=proposal, verdict=verdict)
            continue

        yield _ask(index, proposal, verdict, answer_from, session, on_progress)
        asked += 1


@dataclass(frozen=True, slots=True)
class Gathered:
    """What a run decided before it stopped, and whether it stopped.

    Two facts rather than one, because they answer different questions
    and the reader has to be able to ask both. ``attempts`` is the
    measurement, and a partial measurement is worth having: an evening's
    work that ends in a Ctrl-C has still measured everything it managed
    to do. ``finished`` says whether it is the whole run, and a report
    that printed a yield without it would be claiming to have measured
    the branch when it measured the part of it somebody had time for.
    """

    attempts: list[Attempt] = field(default_factory=list)
    finished: bool = True


def gather(
    attempts: Iterator[Attempt],
    on_attempt: Callable[[Attempt], None] | None = None,
) -> Gathered:
    """Take what a run decides, and survive being stopped.

    An interrupt is the ordinary way this ends, not a failure: the run
    is a minute per question over a list nobody has counted yet, and the
    person watching it is the one who decides it has gone on long enough.
    So it is caught rather than allowed to unwind past everything
    gathered — the work before the interrupt is not lost work, and a
    traceback over it would be a way of pretending otherwise.

    Anything *other* than an interrupt is left to propagate. A model
    that is not answering, or a branch that cannot be read, is a fault
    the reader has to see, and quietly reporting a short run as though
    the short run were the plan would hide it.
    """
    gathered: list[Attempt] = []

    try:
        for attempt in attempts:
            gathered.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
    except KeyboardInterrupt:
        return Gathered(attempts=gathered, finished=False)

    return Gathered(attempts=gathered, finished=True)


def _ask(
    index: Index,
    proposal,
    verdict,
    answer_from: Phraser,
    session: Session | None,
    on_progress: Callable[[str], None] | None,
) -> Attempt:
    asked = turn(
        proposal.question,
        index,
        answer_from=answer_from,
        session=session,
        on_progress=on_progress,
    )
    # The turn's own citations, over what the model wrote. Not
    # `citation_report(asked.outcome)`: by the time the outcome exists
    # the guard has already removed every citation that did not check
    # out, so a recomputation measures the guard rather than the model
    # and reports a tool that invents on every question as perfect.
    return Attempt(
        proposal=proposal,
        verdict=verdict,
        outcome=asked.outcome,
        citations=asked.citations,
    )
