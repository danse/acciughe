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
from acciughe.graph import Grounding, Reach
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


def _grounding(
    ask: Ask,
    seeds: list[str],
    evidence: list[tuple[str, str]],
    index: Index,
) -> Grounding | None:
    """What a turn was given, split by how the question reached it.

    A seed is a note the question's own words reached, and everything
    else in the evidence came through a relation. Counting the two says
    what the model was actually handed: a graph can be dense without any
    of its edges pointing anywhere the question needed, and a bound of
    five notes filled by seeds is a walk that never consulted it.

    **Nothing here for a turn that read nothing.** A question that
    matched no note has no evidence and so has no split to report, and
    counting its absent half as notes that came through the graph would
    be a turn that consulted nothing being counted as one that reached
    the graph. `None` for those, and the report leaves them out.

    ``seeds_agree`` compares the two ways of ranking the seeds, the sum
    of what they share and their single strongest shared term, and says
    whether they named the same one first. They can disagree, and when
    they do the ranking chose a note for its breadth over one that
    matched more precisely. Ranked over every seed and not only the ones
    that fitted in the evidence: a seed the bound dropped was still a
    candidate, and whether the ranking would have led with it is a fact
    about the ranking rather than about how many notes there was room
    for.
    """
    if not evidence:
        return None

    seeded = set(seeds)
    in_evidence = {note_id for note_id, _ in evidence}
    seeded_here = seeded & in_evidence

    def first(weigh) -> str:
        return min(seeds, key=lambda n: (-weigh(n), n))

    def by_sum(note_id: str) -> float:
        return ask.worth(note_id, index.note_text(note_id) or "")

    def by_strongest(note_id: str) -> float:
        return ask.strongest(note_id, index.note_text(note_id) or "")

    return Grounding(
        seeds=len(seeded_here),
        through_edges=len(in_evidence - seeded),
        seeds_agree=first(by_sum) == first(by_strongest),
    )


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
    grounding: Grounding | None = None

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


@dataclass(frozen=True, slots=True)
class Ask:
    """What a question is asking for, and what each of its words is worth
    in this branch.

    The two halves travel together because nothing can be ranked against
    a question without both. A note is relevant when it shares a term
    with the question, and *how* relevant depends on how few notes here
    hold that term — a fact about the branch rather than about the word,
    which a caller cannot work out on its own. Held together for the same
    reason `terms()` takes its stopwords as an argument: seeds and
    evidence are ranked by the same rule, and two functions computing
    relevance separately would be free to drift apart, with one of them
    still looking correct on its own.
    """

    terms: frozenset[str]
    weight: dict[str, float]

    def shared(self, note_id: str, text: str) -> frozenset[str]:
        """The question's terms this note holds.

        A note's own name counts among its terms: a question naming a
        note is about that note, whichever way the note is filed. No
        stopword set is passed because ``terms`` here is already the
        branch's, so intersecting against it removes them again.
        """
        return (terms(text) | {_name(note_id)}) & self.terms

    def worth(self, note_id: str, text: str) -> float:
        """What this note has to say about the question, in the branch's
        own units.

        Summed, so breadth counts — but each term is weighed by how rare
        it is here, so breadth of *common* words does not count as much
        as one subject word.

        Zero is ambiguous on its own: it is both a note that shares
        nothing with the question and a note that shares only terms no
        note could be separated by. `shared` tells the two apart, and a
        caller seeding a walk needs to.
        """
        return sum(self.weight.get(term, 0.0) for term in self.shared(note_id, text))

    def strongest(self, note_id: str, text: str) -> float:
        """What the single most distinguishing shared term is worth.

        The other half of `worth`, and the reason the trade between them
        can be measured rather than argued: a note can outrank another on
        the sum while holding a weaker best term than the note it beat,
        which is the whole of the case for ranking on this instead.
        Comparing the two rankings says whether that ever happens on a
        branch, which is not a question the branch's own vocabulary can
        answer in advance.
        """
        return max(
            (self.weight.get(term, 0.0) for term in self.shared(note_id, text)),
            default=0.0,
        )


def _ask_for(question: str, index: Index) -> Ask:
    """The question read against the branch, once per turn.

    Both halves are derived here rather than by the callers that need
    them, so a turn reads the branch's texts once and every ranking in
    it is scored from the same weights.
    """
    stop = index.stopwords()
    return Ask(
        terms=frozenset(terms(question, stop)),
        weight=distinguishing(dict(index.notes()), stop),
    )


def seeds_for(ask: Ask, index: Index) -> list[str]:
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
    if not ask.terms:
        # Every word was one this corpus's own languages say nothing
        # with, so there is nothing to match on. A question made only of
        # them is not about this corpus.
        return []

    scored: list[tuple[float, str]] = []
    for note_id, text in index.notes():
        if ask.shared(note_id, text):
            scored.append((-ask.worth(note_id, text), note_id))
    return [note_id for _worth, note_id in sorted(scored)]


# --- Reaching -----------------------------------------------------------

def _gather(
    index: Index, reach: Reach, ask: Ask
) -> tuple[list[tuple[str, str]], list[str]]:
    """The notes the walk found, best first, and what a bound leaves out.

    **Ordered by what each note has to say about the question, then by
    how near it was.** Both, in that order, and the first is the one that
    matters: a graph is dense — 41 relations a note on the 131-note
    branch this was measured on — so nearly every note reached sits at
    the same depth as the seed, and ordering by depth alone left the tie
    to the note's name. The evidence was then the five alphabetically
    first of 39 seeds, for a question about halving: five diary notes
    sharing only *the* and *with*, while the note actually about halving,
    ranked first among the seeds, was not among them.

    That put the ranking a seed was given out of reach of the model. A
    walk's seeds are where to start and its evidence is what the answer
    is written from; ranking one and not the other means the ranking
    decided the order of a line in the report and nothing else.

    Depth still decides, so a bound drops the furthest of equally
    relevant notes rather than an arbitrary one — but it can only ever
    be a tie-break, and saying so matters. A note holding a word of the
    question *is* a seed by `seeds_for`'s own rule, and a breadth-first
    walk puts every seed at depth zero, so relevance is above zero
    exactly when the depth is zero. Putting relevance first therefore
    ranks what the question is about ahead of how far the walk had to
    come for it, and never the other way round: a note three hops out
    that the question is *about* cannot outrank a note one hop out that
    it is not, because no such note exists to be one hop out and not a
    seed.

    The seed is included even when the question barely touches it: it is
    where the walk started, and offering its neighbours without it would
    hide what the question was about.

    DECISION: relevance before depth. `product.md` says the walk reads
    the graph and is explicit about refusing when the walk is unconnected
    or thin, but not about the order of the notes it hands the model.
    Nearest-first was the order before, on the reasoning that a bound
    should drop the furthest note rather than an arbitrary one — which
    this keeps, as the tie-break, and no longer lets stand as the first
    key.

    A note that shares nothing with the question scores zero and sorts
    last, behind every note that does, however near. That is the honest
    reading: it is in the evidence because the graph reached it, and the
    walk is allowed to say so.
    """
    def rank(note_id: str) -> tuple[float, int, str]:
        text = index.note_text(note_id) or ""
        return (-ask.worth(note_id, text), reach.depth_of(note_id), note_id)

    order = sorted(reach.reached, key=rank)
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
    ask = _ask_for(question, index)
    seeds = seeds_for(ask, index)
    reach = index.graph().reach(seeds)
    beyond = reach.reached - set(seeds)
    evidence, withheld = _gather(index, reach, ask)
    grounding = _grounding(ask, seeds, evidence, index)

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
        grounding=grounding,
    )
