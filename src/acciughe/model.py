"""One model call per turn, and what it takes to make the answer usable.

Two things live here because they are the same problem: getting a
sentence out of a model that is good at copying and bad at shapes.

Measured on this machine, on this model, against real notes:

- Given a JSON schema, it returns JSON. Given a line format, it copies
  the line format's example text back verbatim — so a prompt that shows
  a fillable example is a prompt that returns the example.
- It copies the *granularity* it is told to. Asked for "the exact words"
  it returned `"The"`; asked for "the whole sentence, do not shorten it"
  it returned whole sentences. The instruction has to name the size.
- Citation fields left optional come back empty, every time. Required
  fields come back filled. Under-citing refuses nearly every question, so
  the fields are required and the guard is what makes that safe.
- It fabricates when it cannot answer rather than declining, so the
  citation check is not a measurement here — it is the only thing
  standing between this model and an invented answer.

None of that is repaired at inference time. A turn is one to two minutes
on this hardware, so there is no retry, no second attempt, and no
re-asking: the reply is parsed once and handed to the guard, and a
reply that cannot be parsed becomes a refusal rather than a crash.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from acciughe.agent import Phraser
from acciughe.session import Answer, Citation, Components

DEFAULT_MODEL = "gemma3:270m"
DEFAULT_HOST = "http://127.0.0.1:11434"

# How many notes a reply is asked to cite. Two was measured to be what
# this model will fill, and a third slot is a third thing it can get
# wrong in a way only the guard can catch.
CITATIONS = 2

# Enough for an answer sentence and two quoted sentences, and no more:
# a turn is not latency-bound, and every extra token is a minute of
# waiting on two cores.
_MAX_TOKENS = 400

# DECISION: greedy. A turn that answered differently on a second run
# would make the evaluation a measurement of sampling noise, and this
# turn is the one product.md asks to be measured.
_TEMPERATURE = 0.0

# The exact words this model produces when it is declining rather than
# answering. Kept because it was measured, and because a reply that says
# this while also quoting is contradicting itself.
DECLINED = "not in note"

_RULES = """\
Copy the answer out of the notes above. Use no other knowledge. Do not
paraphrase and do not add anything the notes do not say.
Set "answer" to one whole sentence copied from the notes."""


# --- The shape of a reply -----------------------------------------------

def reply_shape(slots: int = CITATIONS) -> dict[str, Any]:
    """The JSON schema a reply is asked to fit.

    Flat scalars, not an array. Measured: this model fills scalars and
    returns an empty array, and an answer with no citations is a refusal,
    so an array would mean refusing every question.
    """
    properties: dict[str, Any] = {"answer": {"type": "string"}}
    required = ["answer"]
    for n in range(1, slots + 1):
        properties[f"note{n}"] = {"type": "integer"}
        properties[f"quote{n}"] = {"type": "string"}
        required += [f"note{n}", f"quote{n}"]
    return {"type": "object", "properties": properties, "required": required}


def prompt_for(
    question: str, evidence: list[tuple[str, str]], slots: int = CITATIONS
) -> str:
    """The one prompt a turn sends.

    Notes are numbered from one and the model is asked for the number
    rather than the path: asked for a path it returned the note's text
    in the path field, and asked for a number it returns an integer.
    """
    lines = ["NOTES"]
    for number, (path, text) in enumerate(evidence, start=1):
        lines.append(f"[{number}] {path}")
        lines.append(text)
    lines += ["", f"QUESTION: {question}", "", _RULES]

    for n in range(1, slots + 1):
        lines.append(
            f'Set "note{n}" to the number of the note you copied from and '
            f'"quote{n}" to the whole sentence you copied from it, word for '
            f"word. Do not shorten it."
        )
    lines.append("Fill in every one of those.")
    return "\n".join(lines)


# --- Reading a reply ----------------------------------------------------

def _declined(row: dict[str, Any]) -> bool:
    said = str(row.get("answer", ""))
    return said.strip().lower().rstrip(".!") == DECLINED


def parse(
    reply: str, evidence: list[tuple[str, str]], slots: int = CITATIONS
) -> Answer:
    """Turn a reply into an answer with citations, or into nothing.

    Nothing is sanitised on the way through. A note number the model
    invented is recorded as the path it invented, so the guard can see
    and reject it and so a run can be counted; a quote that is not in
    the note is passed through as written. This function decides shape,
    never truth.
    """
    try:
        row = json.loads(reply)
    except (TypeError, ValueError):
        return Answer(text="", evidence=[])
    if not isinstance(row, dict):
        return Answer(text="", evidence=[])

    answer = str(row.get("answer", ""))
    if _declined(row):
        # It says the notes do not answer this while quoting them. One of
        # the two is wrong and there is no way to tell which, so the
        # answer is nothing rather than a guess about which.
        return Answer(text=answer, evidence=[])

    known = [path for path, _text in evidence]
    citations: list[Citation] = []
    for n in range(1, slots + 1):
        quote = row.get(f"quote{n}")
        if not isinstance(quote, str) or not quote.strip():
            continue
        number = row.get(f"note{n}")
        path = _path_for(number, known)
        citations.append(Citation(path=path, quote=quote))

    return Answer(text=answer, evidence=citations)


def _path_for(number: Any, known: list[str]) -> str:
    """The path a note number names, or the number as written.

    Out of range and non-integers are carried through as text so the
    guard rejects them, rather than being quietly dropped: a dropped
    citation is an answer that looks better-supported than it is.
    """
    if isinstance(number, bool) or not isinstance(number, int):
        return str(number)
    if 1 <= number <= len(known):
        return known[number - 1]
    return str(number)


# --- The model ----------------------------------------------------------

_SPACES = re.compile(r"\s+")


def _is_copied(text: str, evidence: list[tuple[str, str]]) -> bool:
    """Whether the sentence is one of the notes, rather than about them.

    Measured: asked a question it can answer, this model copies the
    sentence out of the note. Asked one it cannot, it writes a new one —
    and that new sentence is the part of a reply that no check reaches.
    The quoted spans beside it are still verbatim and still attributable,
    so the turn is not wasted; the sentence is just not shown.

    A second line, not the first. `agent.turn` still guards the citations,
    and a copied sentence whose quote is fabricated is still refused.
    This decides only whether a *traceable* sentence is shown as the
    answer or set aside in favour of the notes it was drawn from.

    Matching is on whole notes rather than on the cited ones, because the
    sentence may legitimately have been assembled from a note the model
    chose not to quote. DECISION: a sentence is an assertion rather than
    a copy when it appears in *no* note shown. Anything finer needs a
    judgement about whether the sentence follows from the spans, which is
    a second model call, and a turn is one to two minutes.
    """
    said = _SPACES.sub(" ", text).strip().rstrip(".")
    if not said:
        return False
    return any(said in _SPACES.sub(" ", body) for _path, body in evidence)


def phrasing(complete: Callable[[str, dict], str]) -> Phraser:
    """An ``answer_from`` for `agent.turn`, over a one-call completion.

    Split from the transport so that the half which is decidable
    offline is decided offline: what is sent, and what is made of what
    comes back. A turn is one call, so a reply is parsed once and never
    asked about again.

    The model is asked for a sentence and gets first refusal on it: one
    the notes contain is returned as an answer, and one they do not is
    set aside in favour of the quoted spans it was drawn from. One call
    either way, because a turn is a minute or two, and asking again
    would spend a second call on exactly the turns that have already
    failed once.
    """

    def answer_from(
        question: str, evidence: list[tuple[str, str]]
    ) -> Answer | Components:
        reply = complete(prompt_for(question, evidence), reply_shape())
        answer = parse(reply, evidence)
        if _is_copied(answer.text, evidence):
            return answer
        return Components(evidence=answer.evidence)

    return answer_from


class Ollama:
    """A local model server, over its HTTP API. No dependencies.

    One request, one reply, and no retry. A turn already takes a minute
    or two, so a second attempt would double that to fix something the
    guard has to reject anyway.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 900.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, schema: dict) -> str:
        """Ask for a completion, and return the reply's text verbatim.

        Errors are raised rather than swallowed. A server that is not
        running has to be visible: an empty reply would be refused by the
        guard like any other bad answer, and a tool that refuses
        everything because its model is missing is worse than one that
        says so.
        """
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": schema,
                "options": {
                    "temperature": _TEMPERATURE,
                    "num_predict": _MAX_TOKENS,
                },
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["message"]["content"]
