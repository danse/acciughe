"""Measuring, rather than assuming.

Encodes the Evaluation section of product.md:

  What matters most is citation correctness: every claim traces to a
  note. Whether an answer is right is not the measure, since judging it
  means already knowing the answer, which is the thing in doubt. The
  set includes questions only search can answer, so the comparison is
  not won by default.

  The graph is measured rather than assumed — its density, how much of
  it is co-occurrence rather than relation, how many notes end up
  connected to nothing.

**No model is consulted anywhere in this module.** The half of "the
graph proposes; the notes decide" that needs a reader is left
unimplemented rather than faked, and the part that can be checked
without one is here, because that is otherwise the part taken on trust.
Anything a model would be asked to judge in this file is absent, and
the absence is deliberate: a measure that guesses is worse than no
measure, since it is believed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from acciughe.index import Index
from acciughe.keyword import KeywordSearch
from acciughe.propose import Proposal
from acciughe.relations import CO_OCCURRENCE, stopwords, words
from acciughe.session import Answer, Components, Refusal

# Whitespace is not a claim. A quote has to appear in the note, but
# insisting on byte equality would report a true citation as fabricated
# over a line break the author did not think about.
_SPACES = re.compile(r"\s+")


def _flatten(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


# --- Citation correctness ----------------------------------------------

@dataclass(frozen=True, slots=True)
class Finding:
    """A citation that did not check out, which one, and why.

    The quote is carried and not just the path because a report that
    cannot say which of a note's citations was wrong cannot be acted on:
    one bad quote in a note leaves its other quotes standing, and dropping
    the note would lose them.
    """

    path: str
    reason: str
    quote: str = ""


@dataclass(frozen=True, slots=True)
class CitationReport:
    """How much of an answer's evidence is really there.

    ``uncorroborated`` is separate from ``correct`` because an answer
    with no citations does not score as a fraction of nothing. It is the
    worst result, and averaging it away with the others would be the
    one way to make an answer that invents its way past the check look
    respectable.
    """

    total: int
    correct: int
    fabricated: list[Finding] = field(default_factory=list)

    @property
    def uncorroborated(self) -> bool:
        return self.total == 0

    @property
    def fraction(self) -> float:
        if not self.total:
            return 0.0
        return self.correct / self.total


def citation_report(answer: Answer | Components, index: Index) -> CitationReport:
    """Check every citation against the notes it claims to come from.

    Whether the answer is *right* is not asked here, and cannot be:
    deciding that means already knowing the answer, which is the thing
    in doubt. What can be checked is whether each claim's quote is
    actually in the note it names — a receipt that either exists or
    does not.
    """
    fabricated: list[Finding] = []
    correct = 0

    for citation in answer.evidence:
        text = index.note_text(citation.path)
        if text is None:
            fabricated.append(
                Finding(
                    citation.path,
                    "cited note does not exist in the corpus",
                    citation.quote,
                )
            )
        elif _flatten(citation.quote) not in _flatten(text):
            fabricated.append(
                Finding(
                    citation.path,
                    "quoted text is not in the cited note",
                    citation.quote,
                )
            )
        else:
            correct += 1

    return CitationReport(
        total=len(answer.evidence), correct=correct, fabricated=fabricated
    )


# --- The question set ---------------------------------------------------

class Standing(Enum):
    """Whether a proposed question belongs in the set at all."""

    KEPT = "kept"
    DROPPED = "dropped"


KEPT = Standing.KEPT
DROPPED = Standing.DROPPED


@dataclass(frozen=True, slots=True)
class Verdict:
    """Why a question stayed or went, and what the search saw.

    The reason is carried rather than implied, because a question set
    assembled by discarding things silently would be impossible to
    argue with, and this one exists to be argued with.
    """

    verdict: Standing
    reason: str
    supported_by: list[str] = field(default_factory=list)
    found_by_search: list[str] = field(default_factory=list)


def question_verdict(
    text: str,
    supported_by: list[str],
    search: KeywordSearch,
    index: Index,
) -> Verdict:
    """Decide whether a proposed question is worth measuring against.

    Three refusals, in the order the spec puts them:

    - Nothing behind it. A question with no note answering it is not a
      hard case, it is an absent one.
    - The notes decide. Every supporting note has to be in the corpus, or
      a structure the reader invented becomes its own proof.
    - Only the hard ones. If search already surfaces every note that
      answers the question, the baseline wins it by default and the
      question tells us nothing about the graph.
    """
    support = list(supported_by)

    if not support:
        return Verdict(
            DROPPED, "no note is behind this question", support
        )

    present = index.note_ids()
    absent = [note_id for note_id in support if note_id not in present]
    if absent:
        return Verdict(
            DROPPED,
            f"{', '.join(absent)} is not in the corpus",
            support,
        )

    found = search.search(text)
    found_set = set(found)
    if set(support) <= found_set:
        return Verdict(
            DROPPED,
            "search already finds every note that answers it",
            support,
            found,
        )

    return Verdict(
        KEPT,
        "search cannot follow this question to the notes that answer it",
        support,
        found,
    )


# --- How much of the graph is a word everybody writes --------------------

# DECISION: half the notes. A term in this many notes cannot tell one
# from another, so any edge it holds is a coincidence of vocabulary
# rather than a relation. Half rather than nine-tenths because a term
# in ninety per cent of notes is usually the subject of the corpus, and
# dropping the subject of a corpus to tidy its graph would be a cure
# worse than the complaint.
UBIQUITY = 0.5

_NAMED = 12


@dataclass(frozen=True, slots=True)
class Ubiquity:
    """How much of the graph is held together by words that say nothing.

    ``terms`` are the words themselves, most widespread first, because a
    count of "this much of your graph is noise" is not actionable and a
    list of the words is: every one of them is a word the stopword list
    should have known about, and naming them is how a reader finds out
    which language the list is missing.
    """

    notes: int
    edges: int
    held_by_these_alone: int
    share: float
    terms: list[tuple[str, int]]


def ubiquity(
    index: Index, share: float = UBIQUITY, floor: int = 0
) -> Ubiquity:
    """The co-occurrence edges that no word in them actually supports.

    An edge between two notes is real if they share a term that
    distinguishes one from another. An edge whose *only* shared terms
    are the ones nearly every note has is an artefact of everybody
    writing the same function words, and it costs the same to the walk
    as a real one: the graph looks connected, the walk reaches
    everywhere, and nothing was found.

    Counted over co-occurrence only. A link edge is somebody having
    written `[[this]]`, which no vocabulary can fake.

    Computed from the notes rather than kept in the store, because it is
    a measurement of the corpus and not part of the derivation: the same
    graph measured on a corpus with one note removed is a different
    measurement, and nothing here is cached against a timestamp.
    """
    per_note = {
        note_id: words(text) for note_id, text in index.notes()
    }
    counts: Counter[str] = Counter()
    for found in per_note.values():
        counts.update(found)

    # No floor by default, which is deliberate and different from
    # `relations.stopwords`. A derivation has to avoid being wrong, and
    # over a handful of notes a count separates nothing; a measurement has
    # only to be true about what is there, and on a small corpus naming
    # the words everybody writes is the entire point of looking. `floor`
    # asks for the derivation's own view instead.
    everywhere = (
        stopwords(per_note, share=share, floor=floor) if floor
        else frozenset(
            word for word, seen in counts.items()
            if seen >= max(2, share * len(per_note))
        )
    )

    held = 0
    for a, b, kind, _weight in index.edges():
        if kind != CO_OCCURRENCE:
            continue
        shared = per_note.get(a, set()) & per_note.get(b, set())
        if shared and not (shared - everywhere):
            held += 1

    co = sum(
        1 for _a, _b, kind, _w in index.edges() if kind == CO_OCCURRENCE
    )
    widest = sorted(
        ((word, seen) for word, seen in counts.items() if word in everywhere),
        key=lambda pair: (-pair[1], pair[0]),
    )

    return Ubiquity(
        notes=len(per_note),
        edges=co,
        held_by_these_alone=held,
        share=(held / co) if co else 0.0,
        terms=widest[:_NAMED],
    )


# --- The shape of the graph --------------------------------------------

@dataclass(frozen=True, slots=True)
class GraphMetrics:
    """What the graph looks like, counted rather than judged.

    ``pairs`` is distinct from ``edges`` because two stages can relate
    the same two notes, and a note connected to another note is
    connected once however many reasons it had.
    """

    notes: int
    edges: int
    pairs: int
    density: float
    by_kind: dict[str, int]
    connected: int
    unconnected: int
    unconnected_notes: list[str] = field(default_factory=list)


def graph_metrics(index: Index) -> GraphMetrics:
    """Measure the graph product.md names: density, stage, and orphans.

    Every edge is kind-tagged, so "how much of it is co-occurrence
    rather than relation" is a count rather than a matter of opinion.
    """
    rows = index.edges()
    note_ids = index.note_ids()

    pairs: set[frozenset[str]] = set()
    linked: set[str] = set()
    for a, b, _kind, _weight in rows:
        pairs.add(frozenset((a, b)))
        linked.update((a, b))

    unconnected_notes = sorted(note_ids - linked)
    count = len(note_ids)

    return GraphMetrics(
        notes=count,
        edges=len(rows),
        pairs=len(pairs),
        density=(2 * len(pairs) / count) if count else 0.0,
        by_kind=index.edges_by_kind(),
        connected=len(linked & note_ids),
        unconnected=len(unconnected_notes),
        unconnected_notes=unconnected_notes,
    )


# --- The report ---------------------------------------------------------

# DECISION: ten questions before the figures are a comparison rather than
# a description of one run. Two is plainly not, and the reason is not
# sample size in the abstract: a citation fraction over a handful of
# questions is decided by one turn. One model turn writing one invented
# quote out of two citations is the difference between 100% and 50%, and
# a report that printed the percentage without saying so would be
# reporting a single turn as a property of the tool.
#
# Ten is not a number anything was measured to produce — nothing has been
# run yet — and it is chosen so that one turn is a tenth of the result
# rather than a third of it. It is a floor on *reading* the numbers, not
# a gate on producing them: a report over four questions reports them, and
# says in the same breath that four is not a comparison. Hiding them
# would be the other kind of dishonesty, since a reader who was not shown
# what happened cannot tell silence from a result.
MIN_COMPARISON = 10


@dataclass(frozen=True, slots=True)
class Dropped:
    """A question the verdict would not measure against, and why.

    Kept rather than counted, because a question set assembled by
    discarding things cannot be argued with, and this one exists to be
    argued with: a reader who thinks a question was dropped for the wrong
    reason needs to be able to see which reason was used.
    """

    question: str
    reason: str
    found_by_search: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One proposed question, the verdict on it, and what came of asking it.

    An attempt with no ``outcome`` is a question that was not asked, and
    the verdict says which kind of not-asked it is: a question the
    verdict refused is not worth measuring at all, while a kept question
    with no outcome is one the run stopped short of. Both are here
    rather than counted apart, because a yield is only meaningful
    against the number of questions that could have gone into it — and
    deriving one from the other is what stops the two disagreeing.

    ``citations`` is the receipt for **what the model wrote**, not for
    what the reader was shown. The two come apart at the guard: a turn
    whose citations do not all check out falls back to the quotations
    that do, or refuses outright, and everything the model invented has
    already gone by the time the outcome exists. Counting the outcome
    instead would report a tool that invents a quote on every question as
    having perfect citation correctness — a score produced entirely by
    the guard that was supposed to be measured, which is the one thing a
    measure of citation correctness cannot be.

    None where the model was never called at all, which is a different
    fact from a model that was called and cited nothing: the first is a
    turn that found nothing to answer from, the second is a turn that
    found notes and then claimed them with no receipt.
    """

    proposal: Proposal
    verdict: Verdict
    outcome: Answer | Refusal | Components | None = None
    citations: CitationReport | None = None

    def __post_init__(self) -> None:
        """A question the verdict dropped cannot also have been asked.

        The two states are opposites and conflating them is easy, because
        both end with the report holding a ``Refusal``: a question no note
        answered, and a question the tool declined to answer, look the
        same in a list and are not the same fact at all. The first says
        the corpus could not answer; the second says the tool chose not
        to, having found notes it could not turn into a sentence.

        Refused here rather than reconciled, so that putting one in
        requires deciding to and not merely failing to notice.
        """
        if self.verdict.verdict is DROPPED and self.outcome is not None:
            raise ValueError(
                "a dropped question was never asked, so it has no outcome; "
                "a question the verdict refused is not a turn that refused"
            )


@dataclass(frozen=True, slots=True)
class Report:
    """What a run of the evaluation found, and how little of it is a result.

    Every count is here that a reader needs to not be misled, and the two
    that cannot be separated are ``answers`` and ``citations``. Citation
    correctness measured alone is maximised by refusing every question:
    a tool that never answers has no wrong citations, and would report a
    perfect score. So the rate at which answers are produced is not a
    second finding beside the correctness, it is what the correctness is
    counted over, and a report that printed one without the other would be
    reporting the shape of the number rather than the number.

    ``kept`` is the yield: how many proposals were worth measuring at
    all. It is here because a report of nine answers drawn from nine
    hundred proposals is a different thing from a report of nine answers
    drawn from twelve, and the reader cannot see the difference without it.

    ``unasked`` is the other end of that, and is *derived* rather than
    counted beside: it is the kept questions that carry no outcome, so
    the number of questions worth measuring and the number of them that
    were asked cannot drift apart and print a yield of nine over a run
    that asked two. A turn takes a minute or two, so a run over a real
    branch has to stop somewhere, and a stop that is not counted is a
    yield presented as though it were the whole.
    """

    proposed: int = 0
    kept: int = 0
    dropped: list[Dropped] = field(default_factory=list)
    answers: int = 0
    parts: int = 0
    refusals: int = 0
    citations: int = 0
    correct: int = 0
    uncorroborated: int = 0
    fabricated: list[Finding] = field(default_factory=list)

    @property
    def measured(self) -> int:
        """Questions a turn was actually run against."""
        return self.answers + self.parts + self.refusals

    @property
    def unasked(self) -> int:
        """Questions the verdict kept and the run never reached.

        Never negative, and never a number passed in: a dropped question
        is the only one that cannot have an outcome, so the measured
        turns are always a subset of the kept questions.
        """
        return self.kept - self.measured

    @property
    def answer_rate(self) -> float:
        """How often a turn produced a sentence, over the turns run.

        An answer and a set of parts are counted apart. Both are something
        shown to the reader, and averaging them would make a run of
        fallbacks look like a run of answers, which is the one thing the
        two were separated to prevent.
        """
        return self.answers / self.measured if self.measured else 0.0

    @property
    def citation_rate(self) -> float:
        """How much of what was cited is in the note it names.

        Zero when nothing was cited, which is the worst result and not an
        absent one: it is why ``uncorroborated`` is counted rather than
        left to show up as a division.
        """
        return self.correct / self.citations if self.citations else 0.0

    @property
    def is_a_comparison(self) -> bool:
        """Whether the figures can be read as a result rather than a run."""
        return self.kept >= MIN_COMPARISON


def report(attempts: Iterable[Attempt]) -> Report:
    """What a run of the evaluation found.

    A pure count over what was attempted. It asks for no corpus and consults
    no model, so the one thing this project decided must never happen
    cannot happen here: there is nothing in this function that could
    guess. Whether the run was *worth* running is not decided here either
    — that is ``is_a_comparison``, and it is carried rather than applied,
    so a report over four questions is still a report and the reader is
    told what it is instead of being handed nothing.

    Any iterable of attempts, and taken as it arrives. ``trial.run``
    hands them over one at a time so that a run stopped half way still
    has what it decided, and a function that wanted a list first would
    be asking the caller to defeat that.

    Every number here is counted off the same list, and nothing is passed
    in beside it. That is not tidiness: the proposed count, the yield and
    the number of questions a run stopped short of are three views of one
    set, and three numbers supplied from three places are three numbers
    that can disagree. A count of the questions a run had not reached,
    added to the length of the list and set on the report as well, was
    the shape this replaced, and the two were easy to get right
    separately and wrong together.
    """
    attempts = list(attempts)
    checked = [a for a in attempts if a.citations is not None]

    return Report(
        proposed=len(attempts),
        kept=sum(1 for a in attempts if a.verdict.verdict is KEPT),
        dropped=[
            Dropped(a.proposal.question, a.verdict.reason, a.verdict.found_by_search)
            for a in attempts
            if a.verdict.verdict is DROPPED
        ],
        answers=sum(1 for a in attempts if isinstance(a.outcome, Answer)),
        parts=sum(1 for a in attempts if isinstance(a.outcome, Components)),
        refusals=sum(1 for a in attempts if isinstance(a.outcome, Refusal)),
        citations=sum(a.citations.total for a in checked),
        correct=sum(a.citations.correct for a in checked),
        uncorroborated=sum(1 for a in checked if a.citations.uncorroborated),
        fabricated=[f for a in checked for f in a.citations.fabricated],
    )
