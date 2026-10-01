"""Questions the graph proposes, taken from the corpus rather than written.

product.md:

  An agent walking the graph proposes the questions, so they come from
  the corpus rather than from memory.

A note is not a source of statements about its subject. It is a source of
a person asking something about it, and that is what this walks the graph
to find: a question the user wrote down, and the notes the graph says
answer it. The question is lifted verbatim, because a person asking
about their own notes is not writing in the notes' vocabulary — which is
the only reason a question from the corpus can be one that search fails
to follow.

**No model composes these, and one was measured rather than assumed.**
Asking one to write a question that needs notes the words cannot reach
was tried five times across the three models on this machine
(`gemma3:270m`, `llama3.2:1b`, `qwen2.5:0.5b`): every one returned a
question about a single note, in that note's own words, which is a
question the baseline answers as well as this can. A question has to be
a paraphrase to be hard for search, and paraphrase is exactly what a
model this size does not do — it copies, which is what the answering
path needs and no help at all here. So the question is not written here.

Lifting is not a way of avoiding the harder half of the problem. What a
proposal claims is only that these notes are the ones to look in; whether
they answer the question, and whether search could have found them
without the walk, is ``evaluation.question_verdict``'s to decide. The
graph proposes; the notes decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from acciughe.index import Index

# A question is a span that ends in a question mark, and nothing else
# qualifies. Headings that name a subject without asking anything — an
# "Open questions" list heading, a section title — are not questions, and
# treating them as ones proposes a question the user never asked.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Markdown that frames a line rather than being part of what it says. A
# user asking the question would not type the hash or the bullet, and
# leaving them in makes the proposed question something nobody wrote.
#
# Applied twice on purpose: once to the line, which is where a heading or
# a bullet usually sits, and once to each span the sentence split leaves
# behind. The second application is the one that was missing — a bullet
# that follows a sentence end *on the same line* is not at the start of
# anything the line split saw, and it survived into the proposal. On a
# real branch it proposed `- quanti giorni saranno passati ?` from a
# diary, a question with a list marker in it, and the marker is the part
# the reader did not write.
#
# A run of markers is one marker, because a corpus may hold code as well
# as prose and `--` opens a line comment in Haskell. The trailing
# lookahead is what keeps this off a number: `-5 degrees?` starts with a
# marker character and is not framed, and a pattern that stripped it
# would rewrite the sign out of a question nobody wrote that way.
_FRAMING = re.compile(r"^(?:[-*+]+[ \t]*|[#]+[ \t]*)+(?=[ \t]|$)")

# A question mark is not a question. A span holding nothing a person
# wrote a word in -- a bare "??" where a sentence used to be -- is
# punctuation the corpus did not ask anything with, and proposing it
# spends a turn of the evaluation on it. What counts as a word is any
# character a word is made of, not a letter: `\\w` is Unicode by
# default, so this asks the same question of an Italian, a German and a
# Hungarian note, which a corpus of any one language would get wrong.
_WORD = re.compile(r"\w")

# DECISION: how far the walk goes out from a question. A question is a
# span of notes, not a component: at no bound a proposal is supported by
# most of the corpus, every verdict keeps it, and what is being measured
# stops being whether the walk reached the right notes. Two hops is a
# question that reaches past the note it was asked in and stops while
# the answers are still near.
DEFAULT_DEPTH = 2

# DECISION: how many questions one note may put into the set. Measured on
# a real branch of 131 notes: the questions spread over 32 of them, and
# the largest contributor -- an 86k note of continuous writing -- held 35
# of 105, a third of the whole set. Nothing is wrong with any one of
# those 35, and that is the problem: a measurement in which a third of
# the questions come from one note is a measurement of that note. The
# figure being read would move when that note grew and not at all when
# the graph changed.
#
# A cap rather than a length floor, because a floor does not fix it. A
# floor at 3, 4 and 5 words all leave that note holding a third or more
# of the set: its questions are the contentful ones, so they are the last
# to be cut. And no floor at all, because a one-word question is a real
# one -- "stress?" and "katrin?" are someone asking about their own note,
# and the answer to whether it is answerable is the guard's to give at
# turn time, not a word count's to give here.
#
# Three is a share, not a length: at 3 the same branch yields 55
# questions over the same 32 notes with no note above 5%, so the cost is
# the walks this repeats rather than the questions it drops. The first
# three a note asks, in the order it asks them, so nothing is ranked by
# a proxy for how good a question looks.
DEFAULT_PER_NOTE = 3


def questions_in(text: str) -> list[str]:
    """The questions a note asks, in the order it asks them.

    A line is a question if a span of it ends in `?`. Splitting on
    sentence ends first is what keeps "See [[queue]] and the border
    printer jam?" from being proposed whole, with the link and the
    statement in front of the question.

    The span also has to hold a word. Ending in a question mark is what
    makes it a question; a span with nothing in it but punctuation asks
    nothing, and `_WORD` is what says so without knowing the language.
    """
    found: list[str] = []
    for line in text.splitlines():
        framed = _FRAMING.sub("", line).strip()
        if not framed:
            continue
        for span in _SENTENCE.split(framed):
            span = _FRAMING.sub("", span).strip()
            if not span.endswith("?") or not _WORD.search(span):
                continue
            found.append(span)
    return found


@dataclass(frozen=True, slots=True)
class Proposal:
    """A question found in a note, and the notes the walk says answer it.

    ``answers`` is a claim about where to look, not a claim that the notes
    answer: nothing here has read the notes, and the verdict is what
    turns a proposal into a question worth measuring against.
    """

    question: str
    asked_in: str
    answers: list[str]


def proposals(
    index: Index,
    *,
    max_depth: int = DEFAULT_DEPTH,
    per_note: int | None = DEFAULT_PER_NOTE,
    limit: int | None = None,
) -> list[Proposal]:
    """Walk the graph, and propose the questions the notes already ask.

    Reads the store as it stands and refreshes nothing, for the reason
    ``Index.graph`` gives: the caller decides when the branch is read, so
    that a walk is not made against a graph that moved underneath it.

    ``per_note`` bounds how many of one note's questions the set holds,
    for the reason ``DEFAULT_PER_NOTE`` gives. ``None`` is no bound, which
    is what a caller asking for every question in the branch wants.
    """
    graph = index.graph()
    found: list[Proposal] = []

    for note_id in sorted(index.note_ids()):
        text = index.note_text(note_id)
        if text is None:
            continue
        asked = questions_in(text)[:per_note]
        if not asked:
            continue

        reached = graph.reach([note_id], max_depth).reached
        answers = sorted(reached - {note_id})
        if not answers:
            # The graph noticed nothing here, and a question with nowhere
            # to look is the case product.md says this is not for.
            continue

        for question in asked:
            found.append(
                Proposal(question=question, asked_in=note_id, answers=answers)
            )
            if limit is not None and len(found) >= limit:
                return found

    return found
