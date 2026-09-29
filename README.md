# Acciughe

Question-answering across a corpus of notes. An agent runs on your
device, walking a derived graph of the relations between them, and
answers with the notes behind every claim — or refuses cleanly when the
graph does not support one.

**Read [`product.md`](product.md).** It is the spec, and it is
authoritative over the code. This file does not restate it.

## Development

```sh
uv sync
uv run pytest
```

Python 3.11+. No runtime dependencies yet.

## Layout

| Module | What it is |
| --- | --- |
| `corpus.py` | Walking a branch, and deciding what a note is |
| `index.py` | The derived graph, the cheap recheck, and `graph()` — the seam to the walk |
| `relations.py` | Staged edge derivation, every edge tagged with its kind |
| `graph.py` | Reach, cycles, and showing the work behind a walk |
| `session.py` | A conversation, and the typed outcomes of a turn: an answer, a clean refusal, or the parts the walk found |
| `agent.py` | A question, a walk, and the answer or clean refusal that comes of it |
| `model.py` | One call to a local model per turn, and the parsing that makes its answer usable |
| `keyword.py` | Plain keyword search: the baseline the graph is compared against |
| `propose.py` | Questions the graph proposes, lifted from the corpus rather than written |
| `evaluation.py` | Citation correctness, the question-set rule, the graph metrics, and the report a run is counted into |
| `trial.py` | Running the evaluation: propose, judge, ask, keep the receipts |
| `render.py` | Every rendering: a turn, a read, the graph's measurements, a run's report |
| `cli.py` | The commands, and where a branch's graph and conversations are kept |

The tests are the specification for behaviour. Each names the sentence
of `product.md` it pins; each decision the spec does not make is marked
`DECISION:` with its reasoning.

## Agenda

What is built, what is left, and what is still undecided — in
[`agenda.md`](agenda.md).
