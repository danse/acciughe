# Agenda

What is still undecided. `README.md` says what the project is and where
the code is; `product.md` says what it is required to be. What is decided
is not listed here — the code says so, and the tests say so more
precisely. There is no longer anything to build that is not asked for by
one of the two, so there is no `Remaining`; an item leaves this file when
it is decided rather than moving to a list of what was done.

## Not decided

- **What `_SHARE` should be, now that a real branch has said it cannot
  stand.** Measured on 131 notes in Italian and English: the most
  widespread word is in 50 of them, 38%, and no word reaches half — let
  alone nine tenths. So `stopwords()` returns an empty set, and it did so
  quietly: the derivation carried on with every function word in the
  vocabulary as a term, and 2730 of the branch's 8515 possible pairs came
  out related. Seeds were then ranked by how many such words a note shared
  with the question, so "why did the press jam?" started at a note about
  somebody's history.

  Ranking by `distinguishing` fixed the seeding without touching the
  threshold — a term's weight is the log of how few notes hold it, so the
  corpus supplies the numbers and no cutoff is chosen. It does not touch
  the graph, where 32% of all pairs being related is still the shape the
  branch has, and the noise line still cannot be measured.

  The same weighting now ranks the evidence, which is what makes it a fix
  rather than a preference. It was applied to the seeds first and thrown
  away one line later: `_gather` ordered what the model reads by `(depth,
  name)`, every seed sits at depth zero, and so the evidence was the five
  alphabetically first of the seeds. On this branch a question about
  halving was answered from five `anni/` diary notes sharing only *the*
  and *with*, while the note actually about halving — first among the
  seeds — was read by nobody. Ranked, it is first.

  It now has a command that shows it. `acciughe summarise` ends with the
  noise line, and on this branch that line is the one that reports no
  word reaching half the notes and says the graph may be held together by
  vocabulary everybody shares. So the threshold is no longer a judgement
  buried in a derivation — it is the last line of a description a reader
  runs, and the numbers above it are qualified by it on the same screen.
  Whoever decides `_SHARE` now has somewhere to see what it cost.

  No cliff exists to put a threshold on: ranking terms by how widespread
  they are, and by how many pairs they alone relate, give the *same*
  ordering, decaying smoothly. Density against the share, measured, is
  2730 edges at 0.90 and 0.50, 1832 at 0.15, 793 at 0.05. Every one of
  those is a judgement rather than a reading, which is why the threshold
  was left alone rather than set from this. What settles it is a reader
  saying what they want a graph of theirs to look like.

- **Whether a breadth of shared words should outrank one rare word.**
  Left over from the same work and visible in the measurements: a note is
  scored by the *sum* of its shared terms' weights, so a note matching
  `did`, `the` and `why` (8.40) beats a note that actually contains
  `press` (6.17). Taking the strongest single term instead puts `press`
  first, and reads better on the case where the question names something
  that is there. It throws away breadth, so a question naming three parts
  of one subject would seed on whichever note mentioned one in passing.

  The sum is in place, for the seeds and for the evidence alike, and the
  trade is measured rather than guessed at. Frequency alone cannot settle
  it, because `why` is rare in a corpus of diaries and programming notes
  and rare is all frequency knows. What has changed since this was
  written is that the ranking now reaches the notes the model reads,
  which makes the trade worth more than it looked: it is not only which
  note the walk starts from.

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
