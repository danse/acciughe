# AGENTS.md

`product.md` is the spec. Read it before starting work on anything
here — it defines what this project is, and nothing in this file
repeats it, on purpose. One source of truth, nothing to drift.

It is authoritative over the code: if they disagree, the code is wrong.

If a task needs a decision `product.md` doesn't make, say so rather
than picking one quietly. The tests are the specification for
behaviour: they are the record of what was decided.

Verify behaviour by extending the tests, not with throwaway ones. A
scratch script under `/tmp`, a shell loop that edits the source to
watch a test fail, a one-off `python -c` that probes a function — none
of these are tests, and none of them survive. If a behaviour is worth
checking, it is worth a test in `tests/` that stays and is named after
the sentence of `product.md` it pins. When a guard in the source has no
test, the fix is to write that test, not to probe it by hand.
