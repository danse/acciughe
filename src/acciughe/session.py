"""Sessions: a conversation, and what happens when it refuses.

The behaviour here is specified by the tests in tests/test_session.py.

Two structural choices do most of the work of the refusal rules, and
both are here because the spec asks for them rather than the code
preferring them:

- An ``Answer`` and a ``Refusal`` are different types. The evidence of
  an answer and the nearest notes of a refusal cannot be confused,
  because there is no way to hand one to the other. "A refusal keeps the
  nearest notes, marked as not an answer" is then a property of the
  types rather than a rule a renderer has to remember.
- ``next_step`` is part of the outcome. A refusal is a fork rather than
  a wall, and the fork is a value on the outcome, so the three-way
  decision is made in one place instead of by every caller.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# --- The three kinds of not-knowing -----------------------------------

class RefusalKind(Enum):
    """Which kind of not-knowing this is.

    product.md names all three: absent from the corpus, present but
    unconnected, or thin. They are not degrees of one failure, and they
    are not interchangeable to a reader.
    """

    ABSENT = "absent"
    UNCONNECTED = "unconnected"
    THIN = "thin"


ABSENT = RefusalKind.ABSENT
UNCONNECTED = RefusalKind.UNCONNECTED
THIN = RefusalKind.THIN


class Next(Enum):
    """What the conversation does after this turn.

    The fork. Only absence is a wall; the other two are ways through.
    """

    ANSWERED = "answered"
    END = "end"
    REDIRECT = "redirect"
    ASK = "ask"


ANSWERED = Next.ANSWERED
END = Next.END
REDIRECT = Next.REDIRECT
ASK = Next.ASK


# What each kind of not-knowing does to the search. Absence ends it;
# the other two are forks the search continues through.
_FORK = {
    RefusalKind.ABSENT: Next.END,
    RefusalKind.UNCONNECTED: Next.REDIRECT,
    RefusalKind.THIN: Next.ASK,
}


# --- Outcomes ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Citation:
    """A note behind a claim in an answer."""

    path: str
    quote: str


@dataclass(frozen=True, slots=True)
class NearNote:
    """A note kept alongside a refusal.

    ``is_not_an_answer`` is fixed True and cannot be set otherwise, so
    nearest notes cannot be read as evidence.
    """

    path: str
    why_it_is_near: str
    is_not_an_answer: bool = field(default=True, init=False, repr=False)


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer, and the notes behind it."""

    text: str
    evidence: list[Citation] = field(default_factory=list)
    next_step: Next = Next.ANSWERED


@dataclass(frozen=True, slots=True)
class Components:
    """The notes that bear on a question, shown one at a time.

    Not an answer and not a refusal. The walk reached notes and the model
    read them, but what came back was a sentence the notes do not contain,
    so the sentence is not shown and the notes are.

    **No ``text``, deliberately.** product.md: "It never dresses a partial
    result as an answer." The parts are what the walk found; the joining
    of them is the reader's, and a type that could hold a joining would
    let one back in without anybody deciding to put it there.

    Each citation stands alone — one note, one span — so unlike an answer
    a fabricated one can be dropped without leaving a claim standing on
    nothing. That is what makes this the right fallback and the guard's
    "one bad citation refuses the whole answer" rule the wrong one here.
    """

    evidence: list[Citation] = field(default_factory=list)
    next_step: Next = Next.ANSWERED

    @property
    def is_an_answer(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Refusal:
    """A clean refusal: which kind, the nearest notes, and the fork.

    No ``evidence``, so a refusal has nothing to offer as a claim.
    """

    kind: RefusalKind
    nearest: list[NearNote] = field(default_factory=list)
    redirect_to: str | None = None
    wants: str | None = None

    @property
    def next_step(self) -> Next:
        return _FORK[self.kind]

    @property
    def explanation(self) -> str:
        """What the user is told, in their terms.

        The enum names the condition for the code; this names it for
        the reader, because "this is not in your notes" and "this is
        here but connects to nothing you asked about" are different
        situations and a reader has to be able to tell them apart.
        """
        if self.kind is RefusalKind.ABSENT:
            return "This is not in your notes."
        if self.kind is RefusalKind.UNCONNECTED:
            return (
                "These notes are in your corpus but not connected to anything "
                "you asked about."
            )
        return "There is very little here to answer from."


# --- The conversation --------------------------------------------------

@dataclass(frozen=True, slots=True)
class Message:
    """One turn, as recorded."""

    role: str  # "user" or "assistant"
    content: str
    is_answer: bool = False
    evidence: tuple[tuple[str, str], ...] = ()
    nearest: tuple[str, ...] = ()
    refusal_kind: str | None = None
    next_step: Next = Next.ANSWERED
    redirect_to: str | None = None
    wants: str | None = None


class Session:
    """A conversation: a shallow recording of its messages.

    Append-only and flat. Nothing here derives from the index, so
    rebuilding the graph cannot rewrite what was said, and a session
    outlives the store it was had in a conversation with.
    """

    def __init__(
        self,
        path: Path,
        branch: Path,
        messages: list[Message] | None = None,
        session_id: str | None = None,
    ) -> None:
        # Kept outside the branch, since what it records is note content.
        self.path = Path(path)
        self.branch = Path(branch)
        self.session_id = session_id or uuid.uuid4().hex
        self._log = self.path / f"{self.session_id}.jsonl"
        self._messages: list[Message] = list(messages or [])

    # --- recording ---------------------------------------------------

    def ask(self, question: str) -> None:
        self._append(Message(role="user", content=question))

    def answer(self, answer: Answer) -> None:
        self._append(
            Message(
                role="assistant",
                content=answer.text,
                is_answer=True,
                evidence=tuple((c.path, c.quote) for c in answer.evidence),
                next_step=answer.next_step,
            )
        )

    def components(self, parts: Components) -> None:
        """Record the parts without dressing them as an answer.

        ``is_answer`` is false and ``refusal_kind`` is empty, so a
        reopened session can tell this apart from both: it is not a
        conclusion, and it is not a case of not-knowing — the walk found
        the notes.
        """
        self._append(
            Message(
                role="assistant",
                content="\n".join(f"{c.path}: {c.quote}" for c in parts.evidence),
                is_answer=False,
                evidence=tuple((c.path, c.quote) for c in parts.evidence),
                next_step=parts.next_step,
            )
        )

    def refuse(self, refusal: Refusal) -> None:
        self._append(
            Message(
                role="assistant",
                content=refusal.explanation,
                is_answer=False,
                nearest=tuple(n.path for n in refusal.nearest),
                refusal_kind=refusal.kind.value,
                next_step=refusal.next_step,
                redirect_to=refusal.redirect_to,
                wants=refusal.wants,
            )
        )

    def _append(self, message: Message) -> None:
        self._messages.append(message)
        self.path.mkdir(parents=True, exist_ok=True)
        with self._log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_encode(message)) + "\n")

    # --- reading -----------------------------------------------------

    def messages(self) -> list[Message]:
        return list(self._messages)

    def reopened(self) -> "Session":
        """This session, read back from what was written.

        The same directory, the same file, read the way a later process
        would read it. A session outlives the process that had it.
        """
        rows = [
            _decode(json.loads(line))
            for line in self._log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if self._log.exists() else []
        return Session(
            path=self.path,
            branch=self.branch,
            messages=rows,
            session_id=self.session_id,
        )

    @staticmethod
    def sessions(path: Path) -> list[str]:
        """The conversations held in a directory, oldest first.

        A session is one conversation, and a directory holds several.
        They are addressed by name so that continuing one does not
        disturb another.
        """
        directory = Path(path)
        if not directory.is_dir():
            return []
        return sorted(
            log.stem for log in directory.glob("*.jsonl")
        )

    @staticmethod
    def resume(path: Path, branch: Path, session_id: str) -> "Session":
        """Continue a named conversation, with what was already said."""
        return Session(
            path=path, branch=branch, session_id=session_id
        ).reopened()

    def log_path(self) -> Path:
        return self._log

    def close(self) -> None:
        """Nothing is held open. A session is a file."""

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _encode(message: Message) -> dict:
    row = {
        "role": message.role,
        "content": message.content,
        "is_answer": message.is_answer,
        "evidence": [list(e) for e in message.evidence],
        "nearest": list(message.nearest),
        "refusal_kind": message.refusal_kind,
        "next_step": message.next_step.value,
        "redirect_to": message.redirect_to,
        "wants": message.wants,
    }
    return row


def _decode(row: dict) -> Message:
    return Message(
        role=row["role"],
        content=row["content"],
        is_answer=row["is_answer"],
        evidence=tuple(tuple(e) for e in row["evidence"]),
        nearest=tuple(row["nearest"]),
        refusal_kind=row["refusal_kind"],
        next_step=Next(row["next_step"]),
        redirect_to=row["redirect_to"],
        wants=row["wants"],
    )
