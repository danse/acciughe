# Acciughe

## What it is

Acciughe answers questions across a corpus of notes. It runs on your
device and sends nothing anywhere.

## Problem

Notes are individually findable and collectively unintelligible. Search
finds the note you remember writing; it cannot answer the question that
spans six of them written over two years.

What you actually hold is the relations between notes — the recurring
worry, the abandoned approach you returned to, the project that quietly
became another — and those relations are exactly what a flat folder of
text does not represent.

## Constraints

- **Entirely local.** No network calls, no external services, no
  external party. The model runs on your device.
- **Notes are the source of truth.** They are never rewritten, moved or
  interpreted in place.
- **The graph is derived and disposable.** It is rebuilt from the notes
  and can be deleted at any time without loss.
- **Answers cite their evidence.** Every claim points at the notes behind
  it. When the graph does not support an answer, the command says so
  rather than inventing one.
- **Inspectable.** The structure behind an answer can be examined, not
  merely trusted.

## Approach

*Tentative. This is the current hypothesis, not a settled design.*

Notes are turned into a knowledge graph — entities, relations, clusters
— which is then questioned in plain language. A graph is the current
answer to making cross-note questions answerable, and is expected to be
revised.

An extracted graph is dense and approximate: most of its edges are
co-occurrence rather than genuine relation. The tool is therefore built
to show its working, and to distinguish a real structure from an
artifact of how it was derived.

## Open questions

- **Corpus** — what form does a corpus take? A directory of plain text
  and markdown is the simplest candidate; adapters for particular note
  applications are undecided.
- **Staleness** — what happens when notes change after a build? Refresh
  automatically, warn, or leave it to the user?
- **Visual output** — is the structure ever shown, or only described? A
  subgraph drawn in the terminal, or an export to DOT, would make
  "inspectable" mean something.
- **Refusal** — what does a good refusal look like? Offering the nearest
  relevant notes seems more useful than a bare "I don't know".
- **Evaluation** — what would show this works? A fixed question set
  over a real corpus, judged on whether every claim traces to a note,
  and compared against plain keyword search on the same notes.
- **Non-text notes** — voice and images are out of scope; transcription
  is its own product.

## Deferred

Mechanism — schema, extraction, storage, retrieval, model choice,
incremental updates — belongs in `DESIGN.md`.
