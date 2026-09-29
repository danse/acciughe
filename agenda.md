# Agenda

What is left to build, and what is still undecided. `README.md` says what
the project is and where the code is; `product.md` says what it is
required to be. What is already built is not listed here — the code says
so, and the tests say so more precisely.

## Remaining

1. **Running the evaluation against a real branch.** `acciughe evaluate`
   is built and tested: it walks the graph for questions, judges each one
   against the corpus and against the keyword baseline, asks the ones
   worth measuring, and counts the yield, the answer rate and the citation
   correctness of what the model actually wrote rather than of what
   survived the guard. It has never been run on a branch that has
   questions in it, because no branch anyone here has read has any.

   What is missing is not code. It is a corpus whose notes ask things,
   and the measurement is then over so few questions that the report says
   so itself: below ten kept it prints the figures and refuses to call
   them a comparison. Whether a real branch reaches ten is a fact about
   the corpus. The first run is also the only thing that can say whether
   the refusal rate seen on one live turn — a correct sentence with one
   of its two citations mis-attributed, so the guard dropped both — is a
   rate or a fact about that turn.

   A run is now an evening, and built for that: a receipt is written to
   stderr as it is decided, and stopping it costs the question in flight
   and nothing before it. What is still missing is a branch to point it
   at. The reader is running it on their own corpora.

2. **The report carries `ubiquity`.** `_SHARE` is nine tenths and
   `_MIN_NOTES` is three, and both are judgements about notes nobody here
   has read. A corpus of a few hundred notes is what they will be read
   against, and that much is now known: at 0.9 of three hundred notes a
   function word is found and a subject word is not, where at six notes
   the two sat in the same band and no threshold could separate them. So
   the numbers are defensible at this size and the thing left is to check
   them rather than argue them — `ubiquity` names the words that would
   move them, which is the only way to find out without guessing. It
   hangs off the report rather than being asked for separately, since the
   run is the one thing that produces a corpus worth asking it about.

## Not decided

- **Whether a linking convention should be a rule.** Counting the words
  that say nothing over the branch closed the multilingual fault, and left
  a smaller one: a word like "see", in half the notes because half the
  notes link, is in half the branch and no threshold can separate it from a
  subject, so it stays a term and can hold an edge together. `ubiquity`
  reports it and names it.

  A rule could be had — the word on a link line is not a word about the
  note. But it would be a rule about a habit of writing rather than about
  the notes, and `product.md` asks for measured co-occurrence. Naming it
  is the honest outcome for now; whether that is enough is a question
  about the reader's own notes and not about the code.

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
  need help. *The evaluation is what unblocks it, and the questions it
  needs cannot be written without the reader.*

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
