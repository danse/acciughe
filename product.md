# Acciughe

## What it is

Acciughe answers questions across a corpus of notes. It runs an agent
on your device.

## Problem

Notes are individually findable and collectively unintelligible. Search
finds the note you remember writing; it cannot answer the question that
spans six of them written over two years.

What you actually hold is the relations between notes, and those
relations are exactly what a flat folder of text does not represent.

## Corpus

A corpus is a branch of the filesystem, pointed at directly. It holds
 text files, independently from extensions. What is read is decided by
 whether the content decodes as text. A file is a note.

The branch is walked to its depth, excluding hidden files. Symlinked
directories are not followed. The filename and its modification time
are used, nothing is required of a note.

## Reading

A branch is indexed before it is answered from, and re-read when it no
longer matches. Checking is cheap — the branch is walked and compared
with what was last read — and refreshing only redoes what changed, so
an answer is never given from a graph that does not match the branch.

Both happen unasked, and are reported rather than silent. What read a
graph is recorded with a semantic version, so a reader whose
derivation has changed rebuilds instead of answering from stale
structure. A file that cannot be read as text is reported when the
branch is indexed.

## What can be reached

What the command can answer is bounded by what it can ask of the
graph. Reaching along relations from one note to another is what makes
a question about notes rather than about text; finding what loops back
is what makes a cycle reportable. The same reach answers a question
and shows the work behind it.

## Constraints

- **Local agent.** Agent and model run on your device.
- **Notes are the source of truth.** They are never rewritten, moved or
  interpreted in place.
- **A graph is derived and disposable.** It is rebuilt from the notes
  and can be deleted at any time without loss.
- **Answers cite their evidence.** Every claim points at the notes behind
  it. When a graph does not support an answer, the command says so
  rather than inventing one.
- **Inspectable.** The structure behind an answer can be examined, and
  an examination says what it left out.

## Session

Questions are asked in a conversation, which is a shallow recording of
its messages, independent of the index: rebuilding the graph rewrites
nothing that was said. A session writes nothing into the branch, and
is kept outside it, since what it records is note content. Within it,
a refusal is a fork rather than a wall: what is absent ends the
search, what is present but unconnected redirects the question, what
is thin asks for more.

## Evaluation

The comparison is against plain keyword search on the same branch, a
strong baseline for notes written in your own words. What matters most
is citation correctness: every claim traces to a note. Whether an
answer is right is not the measure, since judging it means already
knowing the answer, which is the thing in doubt. The set includes
questions only search can answer, so the comparison is not won by
default.

An agent walking the graph proposes the questions, so they come from
the corpus rather than from memory. The graph proposes; the notes
decide: a question counts only when the notes support it, or a
structure the reader invented becomes its own proof. This finds
failures of one kind, since a question is asked only if the graph
noticed something.

The graph is measured rather than assumed — its density, how much of
it is co-occurrence rather than relation, how many notes end up
connected to nothing — and measured with the agent that will actually
be used, since a result from a larger model says nothing about a local
one.

## Refusal

When the graph does not support an answer, the command says so, and
says which kind of not-knowing it is: absent from the corpus, present
but unconnected, or thin. It never dresses a partial result as an
answer, and never answers with hedging where a clean refusal would
serve. A refusal keeps the nearest notes, marked as not an answer, and
is inspectable like any other. An answer is given only when the
evidence supports it.
