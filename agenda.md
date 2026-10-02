# Agenda

What is still undecided. `README.md` says what the project is and where
the code is; `product.md` says what it is required to be. What is decided
is not listed here — the code says so, and the tests say so more
precisely. There is no longer anything to build that is not asked for by
one of the two, so there is no `Remaining`; an item leaves this file when
it is decided rather than moving to a list of what was done.

## Not decided

- **Whether showing the parts needs at least two of them.** Measured, on
  a live turn: the model wrote a correct sentence, mis-attributed one of
  its two citations, and lost both. What survived was a verified quote
  from a note the walk had reached that had nothing to do with the
  question. That is honest — a quote shown with its note attached cannot
  mislead, because the irrelevance is visible — but one note is a lookup
  rather than a span, and `MIN_BEYOND` already says two is the threshold
  for the walk. The same floor here would turn that turn into a refusal,
  which shows the same notes as `nearest` and marked as not an answer, so
  nothing would be lost. It is not applied, because it refuses more, and
  the fallback exists because this tool already refuses too much. Which
  of those weighs more is a judgement about how the reader will use it,
  not about the code.

- **Whether a third stage is warranted, and with which model.** The
  kind-tagging exists so this can be measured rather than argued:
  implement the stage, compare it against the keyword baseline, and ship
  it only if it earns its place.

  The argument for embeddings over lexical is that `product.md` calls
  keyword search "a strong baseline for notes written in your own words"
  — which is a warning that lexical may only reach parity, since
  paraphrase across two years of notes is the failure this product exists
  to fix. The candidate is `embeddinggemma` (300M, multilingual):
  `gemma3:270m` cannot serve, since Ollama reports its capability as
  `completion`, not `embedding`.

  The hardware bears on it, and not obviously in the graph's favour:
  SSE4.2 with two cores and no GPU, and little disk to spare. A
  multilingual model is several hundred megabytes and hundreds of
  megabytes again of vectors, on a CPU that will run it slowly. Measuring
  the two stages that exist, first, is therefore the cheaper order of
  operations — and the only one that could show the stage is not needed.
  It is not to be pulled until the evaluation says the existing stages
  need help. *The evaluation has now been run on a real branch, and the
  first run could not say: every figure it had was about the guard and
  the model rather than about the graph. It says now, and the two stages
  are what it measures.*

  Settled either way: **the model identity belongs in the derivation
  version.** Vectors baked into the graph depend on which model produced
  them, so changing the model invalidates the graph exactly as a change
  in derivation does. Without that, a reader would answer from structure
  a model that no longer produced.

- **Whether a citation has to be able to pin its note down.** Measured:
  asked for "the exact words" the model answered `"The"`, which is in the
  note, so the check passed — and the claim behind it had nothing behind
  it. The check catches fabrication and nothing else, and those are two
  different failures.

  The obvious candidate is uniqueness: a quote that appears in exactly one
  of the notes shown says which note it means, and one that appears in
  two does not. But it rejects correct citations too, because notes
  really do repeat boilerplate, and a span that is in both is a true
  citation of either. A minimum length would catch `"The"` and not the
  longer shared span, so it is a different rule with a different cost.
  This changes the primary metric, which was decided and tested, so it is
  not to be changed quietly.

- **Whether to measure connected components.** Not a clustering
  question — a traversal, and free from the edges we already have. It
  answers whether the corpus is one connected body of notes or has
  shattered into islands, which is the failure signal that matters. Not
  committed to; it belongs with the rest of the graph metrics.
