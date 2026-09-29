"""The command.

Until now nothing in this project could be run. Every module is a library
with a seam the tests can reach, and the sum of them had no way to be
met by a person with a question.

**Rendering is elsewhere.** `render.py` turns a `Turn` into text and knows
nothing about arguments, terminals or models; this module is the thin
part that connects them. That split is what makes the output testable at
all, since a turn on this hardware is one to two minutes.

Two decisions in here that are not obvious:

**Nothing is written into the branch.** The index is derived and
disposable, the sessions record note content, and a branch is somebody's
notes — so both live under a state directory outside it. The store is
named after a digest of the branch's resolved path, because one state
directory holding two branches' graphs would answer a question from the
wrong notes, and a silent way to be wrong about your own corpus is the
worst kind.

**Progress goes to stderr and the answer goes to stdout.** A turn pauses
for a minute or two, and a reader needs to know the tool is working; but
`acciughe ask ... | less` should still show the answer and not the
narration of how it got there.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from acciughe.agent import turn
from acciughe.evaluation import Attempt, graph_metrics, report
from acciughe.index import Index
from acciughe.model import DEFAULT_HOST, DEFAULT_MODEL, Ollama, phrasing
from acciughe.render import (
    read as render_read,
    render,
    render_attempt,
    render_metrics,
    render_report,
)
from acciughe.session import Session
from acciughe.trial import gather as trial_gather, run as trial_run

DEFAULT_STATE = Path(
    "~/.local/state/acciughe"
).expanduser()


def store_for(branch: Path, state: Path) -> Path:
    """Where the graph for this branch is kept.

    Named after the branch rather than after the branch's name, because
    two branches can be called ``notes`` and be different directories,
    and a store shared between them would answer from notes that are not
    there.
    """
    digest = hashlib.sha256(str(branch.resolve()).encode()).hexdigest()[:8]
    return state / "graphs" / f"{branch.name}-{digest}.sqlite3"


def sessions_for(branch: Path, state: Path) -> Path:
    """Where the conversations about this branch are kept.

    Under the same digest, because a session records which notes were
    read, and a session listed beside a branch it was not about is a
    session nobody can trust.
    """
    digest = hashlib.sha256(str(branch.resolve()).encode()).hexdigest()[:8]
    return state / "sessions" / digest


# --- The commands -------------------------------------------------------

def _ask(args, out, err) -> int:
    branch = args.branch.resolve()
    if not branch.is_dir():
        err.write(f"not a branch: {branch}\n")
        return 1

    sessions = sessions_for(branch, args.state)
    known = Session.sessions(sessions)
    session = Session(
        path=sessions,
        branch=branch,
        session_id=args.session or (known[-1] if known else None),
    )

    result = turn(
        args.question,
        Index(branch=branch, store_path=store_for(branch, args.state)),
        answer_from=phrasing(
            Ollama(model=args.model, host=args.host).complete
        ),
        session=session,
        on_progress=lambda said: err.write(said + "\n"),
    )
    out.write(render(result) + "\n")
    return 0


def _index(args, out, err) -> int:
    """The read on its own, so it can be asked for without a question.

    The same rendering a question would use, taken from the same
    `index.refresh()`, so what is reported cannot drift from what a turn
    reports about the same branch.
    """
    branch = args.branch.resolve()
    if not branch.is_dir():
        err.write(f"not a branch: {branch}\n")
        return 1

    index = Index(branch=branch, store_path=store_for(branch, args.state))
    out.write(render_read(index.refresh()) + "\n")
    return 0


def _graph(args, out, err) -> int:
    """The graph's own measurements, on a branch that has just been read.

    product.md: "The graph is measured rather than assumed." The three
    things it names are computed by `graph_metrics` and nothing rendered
    them, so a reader could see what had been read and never what it had
    come to.

    The read is reported above the figures rather than suppressed, and
    that is not a preamble: the measurements are of a graph, and a graph
    that did not match the branch would be measuring the wrong one. So
    the line saying the notes are current is what makes the numbers below
    it mean what they appear to mean.
    """
    branch = args.branch.resolve()
    if not branch.is_dir():
        err.write(f"not a branch: {branch}\n")
        return 1

    index = Index(branch=branch, store_path=store_for(branch, args.state))
    out.write(
        render_read(index.refresh()) + "\n" + render_metrics(graph_metrics(index)) + "\n"
    )
    return 0


def _sessions(args, out, err) -> int:
    branch = args.branch.resolve()
    sessions = sessions_for(branch, args.state)
    known = Session.sessions(sessions)
    if not known:
        out.write("no conversations yet\n")
        return 0

    for session_id in known:
        log = sessions / f"{session_id}.jsonl"
        said = sum(
            1
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        out.write(f"{session_id}  {said} message" + ("\n" if said == 1 else "s\n"))
    return 0


def _read_only(index: Index):
    """A turn that only reads, for `acciughe index`.

    Built from `turn` rather than beside it so that what it reports is
    the same read a question would report, not a second implementation
    that could drift. The question is one nothing in a corpus uses, so
    the walk finds no seed and the only thing left to show is the read.
    """
    from acciughe.agent import Turn
    from acciughe.session import RefusalKind, Refusal

    empty = Turn(
        question="", outcome=Refusal(RefusalKind.ABSENT)
    )
    return Turn(
        question=empty.question,
        outcome=Refusal(RefusalKind.ABSENT),
        refresh=index.refresh(),
    )


def _evaluate(args, out, err) -> int:
    """Run the evaluation over this branch and print what it found.

    No session, and deliberately. A session is a conversation the reader
    is having with their notes; a trial is a measurement taken of the
    notes, and putting it in the same log would file a question the tool
    asked itself among the questions a person asked.

    Every receipt is written to stderr as it is decided, and a report
    goes to stdout whatever happens. So a run watched for an hour shows
    what it has found so far, and a run stopped at question three still
    prints the three.
    """
    branch = args.branch.resolve()
    if not branch.is_dir():
        err.write(f"not a branch: {branch}\n")
        return 1

    index = Index(branch=branch, store_path=store_for(branch, args.state))
    index.refresh()

    count = 0

    def said(attempt: Attempt) -> None:
        nonlocal count
        count += 1
        err.write(f"[{count}] {render_attempt(attempt)}\n")
        err.flush()

    gathered = trial_gather(
        trial_run(
            index,
            phrasing(Ollama(model=args.model, host=args.host).complete),
            limit=args.limit,
            on_progress=lambda line: err.write(line + "\n"),
        ),
        on_attempt=said,
    )

    if not gathered.finished:
        err.write(
            f"\nstopped after {count} question{'' if count == 1 else 's'}, "
            "of a run that had not finished. what it decided:\n"
        )
    out.write(render_report(report(gathered.attempts)) + "\n")
    return 0 if gathered.finished else 130


# --- The parser ---------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    given = argparse.ArgumentParser(
        prog="acciughe", description="Ask questions of a branch of notes."
    )
    given.add_argument(
        "--branch",
        type=Path,
        default=Path("."),
        help="the notes to answer from (default: here)",
    )
    given.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help=f"where the graph and conversations are kept (default: {DEFAULT_STATE})",
    )
    given.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"the model server (default: {DEFAULT_HOST})",
    )
    given.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"which model to ask (default: {DEFAULT_MODEL})",
    )
    commands = given.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question", help="asked in your own words")
    ask.add_argument(
        "--session",
        help="which conversation to ask in (default: the most recent)",
    )
    ask.set_defaults(run=_ask)

    commands.add_parser("index", help="read the branch and report").set_defaults(
        run=_index
    )
    commands.add_parser(
        "graph", help="measure the graph: density, stages, and orphans"
    ).set_defaults(run=_graph)
    commands.add_parser(
        "sessions", help="list the conversations about this branch"
    ).set_defaults(run=_sessions)

    evaluate = commands.add_parser(
        "evaluate", help="measure the graph against the questions in your notes"
    )
    evaluate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="ask at most this many questions (the rest are reported, not asked)",
    )
    evaluate.set_defaults(run=_evaluate)

    return given


def main(argv: list[str] | None = None, out=None, err=None) -> int:
    """Run one command.

    `out` and `err` are parameters so that a test can run the whole
    command without a terminal. Neither is a fixture: the defaults are
    the real streams and a caller that wants them replaced has to say so.
    """
    args = parser().parse_args(argv)
    return args.run(args, out or sys.stdout, err or sys.stderr)
