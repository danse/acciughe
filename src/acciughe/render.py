"""What a turn looks like when it is read.

Rendering is a pure function of a `Turn`. Nothing here opens a branch,
starts a model, or writes a byte, which is the only reason the output
shape can be pinned by tests at all: a turn takes one to two minutes on
this hardware, and a test that has to wait for one is a test nobody runs
twice.

**Three outcomes, and none of them may be mistakable for another.** A
reader who cannot tell an answer from a set of notes has been shown
something the spec forbids, and the difference is a sentence against no
sentence, a claim against no claim. So the shape of the output carries
the distinction rather than the wording of it.

The other thing every rendering has to do is report rather than
summarise. product.md: "Inspectable. The structure behind an answer can
be examined, and an examination says what it left out." The walk is
shown, including the notes the evidence bound left out, because an
examination that cannot say what it skipped is an assertion.
"""

from __future__ import annotations

from acciughe.agent import Turn
from acciughe.evaluation import (
    KEPT,
    MIN_COMPARISON,
    Attempt,
    GraphMetrics,
    Profile,
    Report,
    Subject,
)
from acciughe.index import Read
from acciughe.session import (
    Answer,
    Components,
    NearNote,
    Next,
    Refusal,
    RefusalKind,
)

# One column, indented under everything else. Notes are quoted rather
# than echoed: a line of prose that looks like the answer is the thing
# this tool must never print by accident.
_NOTE = "  {path}: \"{quote}\""

_NOT_FOUND = (
    "The model wrote something your notes do not contain, so it is not "
    "shown. These are the notes it was working from."
)
_NOT_AN_ANSWER = "These are notes, not an answer."

_FORK = {
    Next.END: None,
    Next.REDIRECT: "try asking: {ask}",
    Next.ASK: "this would help: {wants}",
    Next.ANSWERED: None,
}

# How many notes that reach nothing are named before the rest are only
# counted. Enough to see whether the orphans are scattered or one clump,
# and few enough that a corpus in trouble can still be read. Named rather
# than hidden either way: the count is always on the line, so a truncated
# list is a shorter list and not a different claim.
ORPHANS_SHOWN = 10

# How many subjects, and how many of the notes most related to, are named
# before the rest are only counted. Same reasoning as `ORPHANS_SHOWN` and
# the same rule about it: the count is on the line either way, so a
# truncated list is a shorter list rather than a different claim. A branch
# can hold three hundred folders and three hundred notes worth naming, and
# a profile that printed all of them would be the note list with a
# heading.
SUBJECTS_SHOWN = 12
RELATED_SHOWN = 10

# How many other subjects one subject is said to touch, per line.
_TOUCHES_SHOWN = 4

# What each derivation stage is called in a sentence. Falling back to the
# kind itself is deliberate rather than a gap: a stage added to the graph
# tomorrow would arrive here unnamed, and a line reading "40 similarity" is
# a line asking the reader what that is, where a line that guessed would be
# a line quietly calling a new kind the wrong thing.
_STAGE = {
    "link": "stated links",
    "cooccurrence": "co-occurrence",
}


def render(turn: Turn) -> str:
    """A turn, as a reader sees it."""
    parts = [read(turn.refresh), _walk(turn), _outcome(turn)]
    return "\n".join(part for part in parts if part)


# --- What the index did -------------------------------------------------

def read(report: Read) -> str:
    """The read on its own.

    Public, and not only because `acciughe index` wants it: "Both happen
    unasked, and are reported rather than silent" is about the read not
    being something a question had to ask for, but a branch that has
    just changed should be able to show what was read without anybody
    having to invent a question to ask.

    Both is the check and the refresh, so a turn that found nothing to do
    says that too — a line that costs six words and is the difference
    between knowing the graph is current and assuming it.
    """
    if report.quiet:
        return "notes are up to date"

    said = [f"read {len(report.read)} note" + _plural(len(report.read))]
    if report.added:
        said.append(f"{len(report.added)} new")
    if report.changed:
        said.append(f"{len(report.changed)} changed")
    if report.removed:
        said.append(f"{len(report.removed)} gone")
    if report.rebuilt:
        said.append("graph rebuilt")
    for unreadable in report.unreadable:
        said.append(f"{unreadable.path} is not a note ({unreadable.reason})")
    for source, target in report.dangling:
        said.append(f"{source} points at {target}, which is not here")

    return "notes: " + ", ".join(said)


# --- The walk -----------------------------------------------------------

def _walk(turn: Turn) -> str:
    """The structure behind the outcome, and what it left out.

    A turn that reached nothing says so rather than showing an empty
    list, because "no seeds" and "no notes beyond the seeds" are the two
    different not-knowings the reader is being told apart.

    A walk that got nowhere past the note the question's own words are in
    says that, rather than printing the seed twice and calling the arrow
    a journey. Reaching exactly where it started *is* the unconnected
    case, and the reader should not have to notice that for themselves.
    """
    if not turn.seeds:
        return "walk: no note in your corpus uses the words of the question"

    seeds = ", ".join(turn.seeds)
    beyond = [path for path, _text in turn.evidence if path not in turn.seeds]
    if not beyond:
        line = f"walk: {seeds}, reaching nothing beyond it"
    else:
        line = f"walk: {seeds} → {', '.join(beyond)}"
    if turn.withheld:
        line += f"\n      left out: {', '.join(turn.withheld)}"
    return line


# --- The outcome --------------------------------------------------------

def _outcome(turn: Turn) -> str:
    outcome = turn.outcome
    if isinstance(outcome, Answer):
        return _answer(outcome)
    if isinstance(outcome, Components):
        return _components(outcome)
    return _refusal(outcome)


def _answer(answer: Answer) -> str:
    """The sentence, then the receipts.

    The quotes are not decoration. product.md: "Answers cite their
    evidence. Every claim points at the notes behind it." A sentence
    with nothing under it is the thing the guard exists to prevent, and
    printing it without its citations would undo the guard at the last
    step.
    """
    lines = [answer.text]
    lines += [
        _NOTE.format(path=c.path, quote=_one_line(c.quote))
        for c in answer.evidence
    ]
    return "\n".join(lines)


def _components(parts: Components) -> str:
    """The notes, and an explicit statement that they are not an answer.

    Both lines are load-bearing. The first says why the sentence is
    missing, so its absence is not read as the tool having found
    nothing; the second says what the reader is holding. product.md:
    "It never dresses a partial result as an answer."
    """
    lines = [_NOT_FOUND, _NOT_AN_ANSWER]
    lines += [
        _NOTE.format(path=c.path, quote=_one_line(c.quote))
        for c in parts.evidence
    ]
    return "\n".join(lines)


def _refusal(refusal: Refusal) -> str:
    """Which kind of not-knowing, the nearest notes, and the way through.

    product.md: "a refusal is a fork rather than a wall: what is absent
    ends the search, what is present but unconnected redirects the
    question, what is thin asks for more." A refusal printed without its
    fork is the wall, so the fork is part of the rendering and not an
    extra the caller may forget.
    """
    lines = [refusal.explanation]
    lines += _nearest(refusal.nearest)

    fork = _FORK[refusal.next_step]
    if fork is not None:
        lines.append(
            fork.format(ask=refusal.redirect_to or "", wants=refusal.wants or "")
        )
    return "\n".join(lines)


def _nearest(notes: list[NearNote]) -> list[str]:
    """The notes a refusal keeps, and why each is being kept.

    The reason is the same for all of them within one turn, which is
    itself worth seeing: a reader told "present but unconnected" can
    check that claim by opening one of them.
    """
    return [
        f"  {n.path}: {n.why_it_is_near}" for n in notes
    ]


# --- Small things -------------------------------------------------------

def _plural(count: int) -> str:
    return "" if count == 1 else "s"


def _one_line(text: str) -> str:
    """A quote on one line, so a note cannot fake a boundary.

    A multi-line quote printed raw would put the next note's path in the
    middle of the previous quote, and a reader checking a citation by
    eye would be reading a boundary the tool invented.
    """
    return " ".join(text.split())


# --- The report ---------------------------------------------------------

def render_attempt(attempt: Attempt) -> str:
    """One line of progress, for a run that takes an hour to finish.

    A turn costs a minute or two on this hardware and a real branch
    proposes enough questions to make the run an evening, so the report
    at the end is not the only thing a person sees. This is what they
    see while it goes, and it carries the one figure worth watching as it
    goes: whether the model is still inventing its quotes. A run whose
    first ten questions all fabricate is a run to stop, and that is only
    visible if the fabrication is on the line.

    The dropped questions are here too, though nothing was asked of
    them. A yield of three out of ninety is a fact about the corpus and
    not about the model, and a reader watching the run cannot see that
    without the ninety.
    """
    verdict = attempt.verdict
    where = f"{attempt.proposal.asked_in}: "

    if verdict.verdict is not KEPT:
        return f"{where}{verdict.reason}"

    if attempt.outcome is None:
        return f"{where}kept, not asked"

    said = f"{where}{_outcome_of(attempt)}"
    if attempt.citations is None:
        return said + ", cited nothing"
    cited = attempt.citations
    if cited.fabricated:
        return said + f", {len(cited.fabricated)} of {cited.total} quotes invented"
    return said + f", {cited.correct} of {cited.total} quotes in the notes"


def _outcome_of(attempt: Attempt) -> str:
    """The outcome, named rather than printed.

    The turn's own rendering is the wrong thing here: a progress line
    that repeats the answer would put a sentence of prose on stderr,
    which is where the walk is written, and a reader watching a long run
    would be skimming them together.
    """
    outcome = attempt.outcome
    if isinstance(outcome, Answer):
        return "answered"
    if isinstance(outcome, Components):
        return "parts, not an answer"
    return f"refused ({outcome.kind.value})"


def render_metrics(metrics: GraphMetrics) -> str:
    """The graph, measured rather than assumed.

    product.md: "The graph is measured rather than assumed — its density,
    how much of it is co-occurrence rather than relation, how many notes
    end up connected to nothing." Those three are computed in
    `evaluation.graph_metrics` and this is where they stop being a number
    in a test and become something a reader can read. It is the same
    measurements in the same units, and no threshold is applied here that
    is not already in the code that produced them.

    **Pairs and edges are both on the line, because the gap between them
    is a finding and not a detail.** An edge is a *reason* two notes are
    related; a pair is two notes that are related at all. A co-occurrence
    edge on a pair you already linked by hand is not extra structure, it
    is the second stage agreeing with the first — and a reader told only
    the edge count would read that agreement as a denser corpus. So the
    count that says how connected the notes are and the counts that say
    what each stage found are both here, and the difference is stated.

    **Density is printed as the average number of relations a note
    has**, which is what it is: the mean degree over distinct pairs. Under
    the word "density" a reader expects a fraction, and mean degree is not
    one — printed as `0.014` on a three-hundred note corpus it would read
    as near-empty, where the same figure spoken as "2.7 relations to a
    note" is a corpus that relates to itself generously.

    **The notes that reach nothing are named, and the rest are counted.**
    product.md: "an examination says what it left out." A count of
    orphans is a finding and the names are what a reader can go and look
    at, but a corpus of three hundred notes can have two hundred orphans
    and a list that long is not a shape. So a floor's worth are named, the
    remainder is counted, and the truncation says so rather than letting
    the list look complete.
    """
    lines = [
        f"{metrics.notes} note{_plural(metrics.notes)}, "
        f"{metrics.pairs} pair{_plural(metrics.pairs)} of them related"
    ]

    if metrics.edges > metrics.pairs:
        repeated = metrics.edges - metrics.pairs
        lines.append(
            f"  {metrics.edges} relations in all, {repeated} of them a second "
            "reason on a pair already related"
        )
    elif metrics.edges:
        lines.append(f"  {metrics.edges} relations in all")

    lines.append(f"  {metrics.density:.1f} relations to a note on average")

    if metrics.by_kind:
        lines.append("  " + ", ".join(
            f"{count} {_STAGE.get(kind, kind)}"
            for kind, count in sorted(metrics.by_kind.items())
        ))

    lines.append(
        f"  {metrics.unconnected} note{_plural(metrics.unconnected)} "
        "connect to nothing"
    )
    for note_id in metrics.unconnected_notes[:ORPHANS_SHOWN]:
        lines.append(f"    {note_id}")
    if len(metrics.unconnected_notes) > ORPHANS_SHOWN:
        left = len(metrics.unconnected_notes) - ORPHANS_SHOWN
        lines.append(f"    and {left} more, which this has not named")

    return "\n".join(lines)


def render_report(report: Report) -> str:
    """What a run of the evaluation found.

    **The correctness figure and the rate it was counted over are on one
    line and cannot be separated.** That is the whole shape of this
    rendering, and it is not a formatting preference: citation correctness
    on its own is maximised by refusing every question, so a run of nine
    refusals would print 100% and a reader who did not know the rate would
    take that as a success. Read in this order — how many questions were
    worth asking, how many were answered, and then how good the answers
    were — no figure on screen can be mistaken for a better result than
    the run behind it was.

    A set too small to compare says so, and says it before the figures
    rather than after: the reader has to know what they are looking at
    before they read it, not be told afterwards that it was too small.
    Nothing is withheld for being small, because a reader who was not
    shown what happened cannot tell silence from a result.
    """
    lines = [_yield_of(report)]

    if not report.is_a_comparison:
        lines.append(
            f"  a set of {report.kept} is not a comparison: below "
            f"{MIN_COMPARISON}, one turn is a large share of the figures below"
        )

    lines.append(_outcomes_of(report))
    lines += _citations_of(report)
    lines += _grounding_of(report)
    return "\n".join(line for line in lines if line)


def _grounding_of(report: Report) -> list[str]:
    """What the walk gave the model, and whether the graph earned it.

    **Last, and the only figures here about the walk rather than the
    model.** Everything above counts turns or receipts; these count what
    the turn was handed, which is the thing no other line here can see. A
    run can cite every quote correctly out of five notes the question was
    never about, and it is the same run that prints 100%.

    The edge share is the one that can settle a question the agenda has
    been carrying, so it is spoken as a share rather than left as two
    numbers to divide. Near zero says the co-occurrence stage did no work
    here and a threshold on the words would take nothing away; most of it
    says the stage is load bearing and the threshold is what decides how
    much of the model's evidence is worth having.

    Said rather than omitted when it is zero, which is the finding: a run
    that read five notes a turn and every one of them matched the
    question's own words is a run where the graph was not consulted, and
    that reads as a null result here rather than as silence.
    """
    if not report.grounded:
        return []

    through = report.read_through_edges
    read = through + report.read_from_questions
    lines = [
        "  nothing the model read came through a relation"
        if not through
        else f"  {_fraction(through / read)} of what the model read came "
        f"through a relation, {through} notes of {read}"
    ]

    turns = len(report.grounded)
    agreed = report.breadth_agreed
    if agreed == turns:
        lines.append(
            "  ranking seeds by the sum of what they share and by their "
            "strongest single word named the same seed every time, so the "
            "sum cost nothing on this run"
        )
    else:
        lines.append(
            "  ranking seeds by the sum of what they share and by their "
            f"strongest single word named the same seed in {agreed} of "
            f"{turns} turns, and the sum is what chose the other "
            f"{turns - agreed}"
        )

    return lines


def render_summary(profile: Profile) -> str:
    """What the corpus holds, counted rather than written.

    product.md: "A corpus can also be described rather than questioned.
    The description is counted from the same graph an answer is walked:
    the subjects it holds, the notes most related to, and which subjects
    touch which others."

    **No sentence here is written about the corpus, only counted.** That
    is the spec's own reason and it is a good one: a description carries
    no citations, so a sentence about what a corpus *is* would be the one
    claim in this tool that nothing could check. Every line below is a
    number a reader can go and verify, and that is also what makes it
    worth having — a corpus too large to read is exactly the one a
    sentence about it would be worth the least.

    **The subjects come with their reach, not just their size.** A count of
    notes is the directory listing the reader already has; what the folder
    *connects to* is the part they cannot see without running anything, and
    it is the same count the graph was built from. So each subject carries
    its notes, the pairs inside it, and the subjects it reaches.

    **Touching is printed both ways, and that is not redundant.** A pair of
    subjects can be very lopsided — 43 notes reaching 2, and 2 notes
    reaching 43 — and a reader who saw only the heavier side would take
    that to be the whole relation. The counts are of pairs, and they will
    not match, because one note's pair is counted once.
    """
    lines = [
        f"{profile.notes} note{_plural(profile.notes)} in "
        f"{len(profile.subjects)} subject{_plural(len(profile.subjects))}"
    ]

    lines += _subjects_of(profile)
    lines += _related_of(profile)
    return "\n".join(line for line in lines if line)


def _subjects_of(profile: Profile) -> list[str]:
    """The subjects, largest first, and what each reaches.

    Largest by notes rather than by reach, because the notes are the thing
    a reader recognises and the reach is the thing they are being told.
    """
    if not profile.subjects:
        return ["  this branch holds no notes"]

    lines = ["  subjects:"]
    for subject in profile.subjects[:SUBJECTS_SHOWN]:
        line = (
            f"    {subject.name}: {subject.notes} note"
            f"{_plural(subject.notes)}, {subject.inside} pair"
            f"{_plural(subject.inside)} among themselves"
        )
        lines.append(line + _touches_of(subject))

    if len(profile.subjects) > SUBJECTS_SHOWN:
        left = len(profile.subjects) - SUBJECTS_SHOWN
        lines.append(f"    and {left} more subject{_plural(left)} not named here")
    return lines


def _touches_of(subject: Subject) -> str:
    """The subjects this one reaches, most pairs first.

    A subject that reaches nothing says so rather than printing nothing,
    because an empty field on a line of numbers reads as a column the
    reader has not been shown yet.
    """
    if not subject.touches:
        return ", reaching no other subject"

    ranked = sorted(subject.touches.items(), key=lambda kv: (-kv[1], kv[0]))
    said = ", ".join(f"{name} {count}" for name, count in ranked[:_TOUCHES_SHOWN])
    if len(ranked) > _TOUCHES_SHOWN:
        left = len(ranked) - _TOUCHES_SHOWN
        said += f", and {left} more it reaches"
    return f", reaching {said}"


def _related_of(profile: Profile) -> list[str]:
    """The notes most connected to others, most first.

    A count of notes reaching nothing is not here: `graph` names those, and
    a profile that repeated them would be a second place to look for one
    fact.
    """
    if not profile.related:
        return ["  no two notes are related, so there is nothing to rank"]

    lines = ["  most related notes:"]
    for note_id, count in profile.related[:RELATED_SHOWN]:
        lines.append(f"    {note_id}: {count} note{_plural(count)}")

    if len(profile.related) > RELATED_SHOWN:
        left = len(profile.related) - RELATED_SHOWN
        lines.append(f"    and {left} more note{_plural(left)} not named here")
    return lines


def _yield_of(report: Report) -> str:
    line = (
        f"{report.kept} of {report.proposed} proposed question"
        f"{_plural(report.proposed)} kept"
    )
    if report.unasked:
        line += (
            f", {report.unasked} judged and not reached"
        )
    return line


def _outcomes_of(report: Report) -> str:
    said = (
        f"answers {report.answers} of {report.measured} asked "
        f"({_percent(report.answer_rate)})"
    )
    if report.parts:
        said += f", {report.parts} shown as parts"
    if report.refusals:
        said += f", {report.refusals} refused"
    return "  " + said


def _citations_of(report: Report) -> list[str]:
    """The receipts, and the ones that did not check out.

    Two absences are being kept apart, and printing one number for both
    would be wrong twice over.

    A refusal cites nothing, because it claimed nothing, and a run of
    refusals has nothing to check. That is not a score of zero — it is the
    absence of one, and saying 0% would report a tool that correctly
    declined to speak as though it had spoken and been wrong.

    An answer that cites nothing *is* the worst result available, and it
    is a claim with no receipt behind it. It gets its own line, so it
    cannot hide inside a fraction whose denominator counted other
    answers' citations.
    """
    if not report.citations:
        lines = ["  nothing was cited, so there is nothing to check"]
    else:
        lines = [
            f"  citations {report.correct} of {report.citations} in the note "
            f"they name ({_percent(report.citation_rate)})"
        ]
    if report.uncorroborated:
        lines.append(
            f"  {report.uncorroborated} answered with nothing behind them, "
            "which is worse than citing something untrue"
        )
    for finding in report.fabricated:
        lines.append(f"  not in the note: {finding.path} — {finding.reason}")
    return lines


def _percent(fraction: float) -> str:
    return f"{round(fraction * 100)}%"


# How a share is spoken when it is one of the few a rule lands on rather
# than a figure that fell out of a count. Half, three quarters and nine
# tenths are the ones; anything else falls back to the number, because a
# strange fraction spelled in words would be read as a precise claim
# about a figure that was not chosen.
_FRACTIONS = {
    0.5: "half",
    0.9: "nine tenths",
    0.75: "three quarters",
}


def _fraction(share: float) -> str:
    return _FRACTIONS.get(share, _percent(share))
