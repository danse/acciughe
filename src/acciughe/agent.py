"""One turn: a question, a walk, and what came of it.

The behaviour here is specified by the tests in tests/test_agent.py.

The load-bearing choice is that the refusal is **structural**. Which kind
of not-knowing this is — absent, unconnected, thin — is a fact about the
graph, readable before anything is generated. So the three forks are
decided without a model, every one of them is reachable from a fixture
rather than from a sampling, and a refused question costs nothing to ask.
On hardware where a turn takes a minute or two, that is the difference
between a tool that is usually right and one that is usually waiting.

What is left for a model is the part that is genuinely not decidable
here: phrasing an answer, and choosing which spans to quote. It arrives
as ``answer_from`` rather than being hard-wired, so nothing in this
module knows what a model is and every path through it is testable
without one.

Two things are kept that a tidier implementation would drop:

- ``withheld``. The walk is unbounded and the evidence is not, so a turn
  is routinely a partial view of what it found. product.md requires an
  examination to say what it left out, and a bound that did not report
  itself would quietly make the graph look smaller than it is.
- ``proposed`` and ``citations``. The guard decides what the reader is
  shown; it does not decide what the model said. Discarding the
  difference would make citation correctness unmeasurable, because every
  answer would then be perfect by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from acciughe.evaluation import CitationReport, citation_report
from acciughe.graph import Reach
from acciughe.index import Index, Read
from acciughe.relations import distinguishing, terms
from acciughe.session import (
    ABSENT,
    THIN,
    UNCONNECTED,
    Answer,
    Components,
    NearNote,
    Next,
    Refusal,
    Session,
)

# DECISION: one note beyond the seed is a lookup and two is a span.
# Below this the walk found a note and nothing to join it to, which is a
# different situation from finding a body of notes that still does not
# answer the question, and the spec counts them apart.
MIN_BEYOND = 2

# DECISION: how many notes reach the model. A turn on this hardware is
# already one to two minutes and holds exactly one model call, so the
# evidence has to fit in that call. The walk itself is not bounded: what
# the bound leaves out is reported rather than dropped.
MAX_EVIDENCE = 5

# Why a note is being kept next to a refusal. The reason is the point:
# "these notes are near" on its own reads as "these are the answer".
_UNCONNECTED_WHY = "your question's words are in this note, and it reaches nothing"
_THIN_WHY = "the walk reached this note and found nothing to reach beyond it"
_UNBACKED_WHY = "the walk reached this note, but the answer did not quote it"

# DECISION: what a thin answer asks for. A refusal that does not name
# the way through it is a wall wearing the costume of a fork.
_UNBACKED_WANTS = "an answer that quotes the notes it rests on"


class Phraser(Protocol):
    """What a turn asks for once the graph has decided there is an answer
    to give: the answer, phrased from the evidence and citing it.

    Or the parts of it, when the model wrote a sentence the notes do not
    contain. Both come from one call, so a turn is not retried to find
    out which one it got.
    """

    def __call__(
        self, question: str, evidence: list[tuple[str, str]]
    ) -> Answer | Components: ...


@dataclass(frozen=True, slots=True)
class Turn:
    """Everything a turn did, and everything it decided.

    ``outcome`` is what the reader is shown. The rest is what the turn
    was shown to reach it, and is kept so the two can be compared.
    """

    question: str
    outcome: Answer | Refusal | Components
    seeds: list[str] = field(default_factory=list)
    evidence: list[tuple[str, str]] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)
    refresh: Read = field(default_factory=Read)
    proposed: Answer | Components | None = None
    citations: CitationReport | None = None

    @property
    def next_step(self) -> Next:
        """What the conversation does next, read off the outcome rather
        than decided a second time."""
        return self.outcome.next_step


# --- Finding somewhere to start ----------------------------------------

def _name(note_id: str) -> str:
    """A note's own name, as the one word it is filed under."""
    base = note_id.rpartition("/")[2]
    return base.rsplit(".", 1)[0].lower() if "." in base else base.lower()


def seeds_for(question: str, index: Index) -> list[str]:
    """The notes to start the walk from, most promising first.

    A note is a seed when it shares at least one distinctive word with
    the question. A seed is where the walk begins rather than evidence
    for an answer, so one word is enough where co-occurrence asks for
    two: that stage derives a weighted relation between two notes, and
    holding a starting point to that standard would drop notes the
    question plainly names.

    **Ranked by how much those shared words distinguish, not by how many
    there are.** `distinguishing` weighs a term by the log of how few
    notes hold it, so a note matching on one subject word outranks a note
    matching on four words every note in the corpus writes. A count got
    this backwards on a corpus in two languages: the words a question and
    a note have most often in common are the ones with the least to say
    about either, and every candidate shares them equally, so the count
    was ordering the seeds by noise.

    Ties break on the note's own name so that the walk is the same
    however the store happens to be ordered.
    """
    stop = index.stopwords()
    asked = terms(question, stop)
    if not asked:
        # Every word was one this corpus writes in every note, so there
        # is nothing to match on. A question made only of them is not
        # about this corpus.
        return []

    notes = dict(index.notes())
    weight = distinguishing(notes, stop)

    scored: list[tuple[float, str]] = []
    for note_id, text in notes.items():
        shared = (terms(text, stop) | {_name(note_id)}) & asked
        if shared:
            scored.append((-sum(weight.get(term, 0.0) for term in shared), note_id))
    return [note_id for _score, note_id in sorted(scored)]


# --- Reaching -----------------------------------------------------------

def _gather(
    index: Index, reach: Reach
) -> tuple[list[tuple[str, str]], list[str]]:
    """The notes the walk found, nearest first, and what a bound leaves
    out.

    Nearest first rather than alphabetical, so a bound drops the furthest
    note rather than an arbitrary one. The seed is included: it is the
    note the question landed on, and offering its neighbours without it
    would hide what the question was actually about.
    """
    order = sorted(reach.reached, key=lambda n: (reach.depth_of(n), n))
    evidence = [(n, index.note_text(n) or "") for n in order[:MAX_EVIDENCE]]
    return evidence, order[MAX_EVIDENCE:]


def _near(paths, why: str) -> list[NearNote]:
    return [NearNote(path=path, why_it_is_near=why) for path in paths]


def _redirect_to(path: str) -> str:
    """DECISION: the fork leads to a question rather than to a path.

    The graph's only offer is the note it found, and a reader told
    "unconnected" with nowhere to go has been handed a dead end and extra
    steps. The note itself is the thing worth asking about next.
    """
    return f"What is in {path}?"


def _wants_more(seeds: list[str]) -> str:
    """DECISION: a thin walk asks for the thing that would have filled
    it, named after what it found — so the reader knows what to write
    toward rather than being told to write more."""
    return f"notes that relate to {', '.join(seeds)}"


def _supported(citations: CitationReport) -> bool:
    """Whether an answer may be given, judged on its receipts alone.

    One bad citation refuses the whole answer. An answer's text was
    written against evidence that turns out not to be there, and there is
    no way to tell which claims the bad quote was carrying; dropping the
    citation would leave its claims standing on nothing, which is the
    partial result the spec forbids dressing as an answer.

    No citations at all is the same case: every claim points at the notes
    behind it, so a claim with nothing behind it is not a claim.
    """
    return citations.total > 0 and not citations.fabricated


def _dropping_fabricated(
    proposed: Answer | Components, citations: CitationReport
) -> list[Citation]:
    """The citations that check out, when some of them do not.

    The mirror image of `_supported`, and reachable from an `Answer` as
    well as from `Components`: what a turn falls back to is the set of
    quotations, and whether the model wrote a sentence over them or not
    changes only whether that sentence is shown.

    Each quotation is one note and one span and asserts nothing beyond
    itself, so the ones that check out are still true after the one that
    does not is gone. An answer has no such property, which is the whole
    reason `_supported` refuses rather than filters: its claims were
    written against the whole of its evidence, and there is no way to
    tell which of them a dropped quote was carrying. Falling back drops
    the claims.

    Matching is by (path, quote) rather than by path alone, because a
    model can mis-attribute — measured, it took a real quote out of one
    note and named another — and dropping every citation of a note
    because one of its quotes was wrong would throw away the ones that
    were right.
    """
    bad = {(f.path, f.quote) for f in citations.fabricated}
    return [c for c in proposed.evidence if (c.path, c.quote) not in bad]


# --- The turn -----------------------------------------------------------

# What a reader is told during the slow part. The note names are here
# because they are the last thing the reader saw before the pause and the
# first thing they will want afterwards; the duration is here because a
# silent terminal and a minute-long turn look identical from the outside.
_SLOW = "  the model is writing; on this hardware that is one to two minutes"


def _reading(evidence: list[tuple[str, str]]) -> str:
    paths = ", ".join(path for path, _text in evidence)
    return (
        f"reading {len(evidence)} note{'' if len(evidence) == 1 else 's'}: "
        f"{paths}\n{_SLOW}"
    )


def turn(
    question: str,
    index: Index,
    *,
    answer_from: Phraser,
    session: Session | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Turn:
    """Ask a question of the graph, and get an answer or a clean refusal.

    The branch is indexed first, unasked, and what that read is reported
    on the turn. ``answer_from`` is called at most once, and only on the
    path where the graph has already established that an answer is
    available.

    ``on_progress`` is called at most once too, and only immediately
    before that call, so a reader is told what is being read during the
    one part of a turn that takes a minute or two. Everything else here
    is fast enough not to need saying. Nothing is reported to it that
    the turn does not then show.
    """
    if session is not None:
        session.ask(question)

    refresh = index.refresh()
    seeds = seeds_for(question, index)
    reach = index.graph().reach(seeds)
    beyond = reach.reached - set(seeds)
    evidence, withheld = _gather(index, reach)

    proposed: Answer | None = None
    citations: CitationReport | None = None

    if not seeds:
        outcome: Answer | Refusal = Refusal(ABSENT)
    elif not beyond:
        outcome = Refusal(
            UNCONNECTED,
            nearest=_near(seeds, _UNCONNECTED_WHY),
            redirect_to=_redirect_to(seeds[0]),
        )
    elif len(beyond) < MIN_BEYOND:
        outcome = Refusal(
            THIN,
            nearest=_near(sorted(reach.reached), _THIN_WHY),
            wants=_wants_more(seeds),
        )
    else:
        if on_progress is not None:
            on_progress(_reading(evidence))
        proposed = answer_from(question, evidence)
        citations = citation_report(proposed, index)
        if _supported(citations):
            outcome = proposed
        else:
            # A failed citation does not end the turn. What the citations
            # have in common is that each is one note and one span, so
            # the ones that check out are still true without the one that
            # does not — and no sentence is being shown, which is what
            # removes the reason an answer has to refuse itself whole.
            kept = _dropping_fabricated(proposed, citations)
            outcome = (
                Components(evidence=kept)
                if kept
                else Refusal(
                    THIN,
                    nearest=_near([path for path, _ in evidence], _UNBACKED_WHY),
                    wants=_UNBACKED_WANTS,
                )
            )

    if session is not None:
        if isinstance(outcome, Answer):
            session.answer(outcome)
        elif isinstance(outcome, Components):
            session.components(outcome)
        else:
            session.refuse(outcome)

    return Turn(
        question=question,
        outcome=outcome,
        seeds=seeds,
        evidence=evidence,
        withheld=withheld,
        refresh=refresh,
        proposed=proposed,
        citations=citations,
    )
