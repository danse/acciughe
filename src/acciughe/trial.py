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

    ``fault`` is what stopped it, when something did: a model server
    that answered and then stopped, or a branch that became unreadable.
    Carried rather than raised, because by the time it arrives there is
    a measurement to keep, and raised rather than reported, because a
    run that stopped is not a run that finished with less in it. `None`
    for the two ordinary endings — a run that finished, and a run the
    reader stopped — which is what tells those apart from a fault.
    """

    attempts: list[Attempt] = field(default_factory=list)
    finished: bool = True
    fault: BaseException | None = None


def gather(
    attempts: Iterator[Attempt],
    on_attempt: Callable[[Attempt], None] | None = None,
) -> Gathered:
    """Take what a run decides, and survive being stopped.

    Two ways this ends that are not the run finishing, and they are not
    the same thing, which is why they are caught differently.

    An interrupt is the ordinary one: a minute a question over a list
    nobody has counted yet, and the person watching is the one who
    decides it has gone on long enough. Nothing is wrong with it, so
    there is nothing to see.

    A fault is not ordinary, and the temptation is to let it unwind past
    everything gathered, which is what this did and what cost a measured
    branch its 33 receipts: the model answered 33 times and then returned
    an `HTTPError` 500, and the distinction the old rule was drawing —
    "a report printed over a run that died of one would be a short run
    presented as though the short run were the plan" — does not hold
    against a run that *says* it stopped short and says why. The report
    renders `finished`, so a fault is not hiding anything; all it was
    doing was throwing away real measurements over an empty traceback.

    So a fault is caught, and carried. Not swallowed: `fault` names it,
    the run is marked unfinished, and the CLI prints it beside the
    report, so a reader is told the model failed rather than left to
    wonder why a branch of 131 notes yielded 33 questions.

    **With nothing gathered, a fault still propagates.** An empty run
    has nothing to report, so there is no measurement to save and no
    partial result to qualify — the only honest output is the exception.
    That is the case the old rule was written for, and it is why the two
    are handled differently rather than one uniformly.
    """
    gathered: list[Attempt] = []

    try:
        for attempt in attempts:
            gathered.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
    except KeyboardInterrupt:
        return Gathered(attempts=gathered, finished=False)
    except Exception as fault:
        if not gathered:
            raise
        return Gathered(attempts=gathered, finished=False, fault=fault)

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
        grounding=asked.grounding,
    )
