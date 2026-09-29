"""Deriving relations from notes.

Staged, cheapest first, and every edge tagged with the stage that
produced it. product.md requires the graph to be measured for "how much
of it is co-occurrence rather than relation"; tagging is what turns that
into a count.

The stages here are pure functions of the notes and consult no model.
That is deliberate: it is what makes a refresh reproducible and what
lets "refreshing only redoes what changed" be true rather than
approximate. A similarity stage will sit behind the same edge type once
an embedding model is chosen.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# The stage an edge came from. Adding a value here changes what the
# graph is, so it belongs in the derivation version.
LINK = "link"
CO_OCCURRENCE = "cooccurrence"
SIMILARITY = "similarity"  # reserved; no backend chosen yet

# A wikilink target, or a relative markdown link target.
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# DECISION: two shared distinctive terms. One is usually "note" or
# "file"; two is the smallest threshold that reliably rejects that.
_MIN_SHARED_TERMS = 2

# DECISION: a word shorter than this carries no distinction. Language
# independent on purpose — it is about letters, not about which language
# a note is written in, and no language of the world spells a
# contentful word in one or two letters.
_MIN_TERM_LENGTH = 3

# DECISION: a term in this share of the notes says nothing about any of
# them, and is dropped. This replaces a fixed list of English stopwords,
# which was patched rather than derived and could not work at all on a
# corpus in another language.
#
# Corpus statistics rather than a fixed list, because whether a word is
# meaningless is a fact about *these* notes, not about a language. "The"
# is the commonest word in English and the most informative one in a
# corpus about typography. A fixed list cannot know that; a count can.
#
# And language independent for the case that forces it: a genuinely mixed
# corpus has no one list to apply. Every note contributes its words and
# the ones everybody wrote fall out on their own.
#
# Nine tenths, and high is the safe direction to be wrong in: a term left
# in costs an edge, a term dropped loses the one relation it held, and
# nothing inside the graph can say which of the two mattered.
#
# The number is high because of what these words are. A function word is
# written in nearly every note — "the" is in all of them or the corpus is
# not written in English — so a share near one finds them all. A content
# word is not, however common its subject: a corpus of one subject has
# that subject in every note, and dropping it at 0.9 is the right answer
# anyway, because a word in every note relates every note to every other
# and the relation it draws is the whole corpus.
#
# Half was tried and measured against six-note fixtures, and at 0.5 both
# are dropped, because at 0.5 a corpus of six notes cannot tell a function
# word from a subject and neither can any threshold. That measurement said
# more about the fixtures than about the number, and it is recorded here
# because the same argument does not apply at the size a real corpus is:
# 0.9 of three hundred notes is two hundred and seventy, and no subject
# word is in two hundred and seventy of somebody's notes. The two
# distributions that overlap at six notes are far apart at three hundred,
# which means the number is no longer the binding constraint — the floor
# below is, and 0.9 is only high enough to be safe.
#
# A parameter, and `evaluation.ubiquity()` measures what a choice costs
# on a real branch rather than taking this comment's word for it.
_SHARE = 0.9

# Below this many notes nothing is dropped, because a count needs a note
# that does *not* share the word to be a count at all. Two notes agree on
# everything or disagree on everything, so "the" and "corpus" are
# indistinguishable at any threshold and guessing deletes the subject.
# Three is the smallest corpus that can distinguish sharing from not.
#
# A floor against a fixture, and negligible at a real corpus: three notes
# is nothing next to the hundreds a branch holds, so at the size this
# runs at the floor is not what stops a word being dropped. It stays
# because the tests are small, and a corpus too small to measure is not
# protected from its own grammar by any threshold.
#
# A parameter, because three is a judgement about somebody else's notes.
_MIN_NOTES = 3

# Letters and digits, in any script. An ASCII pattern does not merely lose
# a word it cannot read — it shortens the one it is in the middle of, so
# "città" becomes "citt" and every other word beginning "citt" with it.
# The same pattern as `keyword.py`, because the baseline and this stage
# read the same notes and the notes are not required to be English.
#
# What this does not fix: a script that does not separate words with
# spaces. Chinese and Japanese come through as one term per clause, which
# is a segmentation problem and not a character-class one.
_WORD = re.compile(r"[^\W_]+")


@dataclass(frozen=True, slots=True)
class Edge:
    """A relation between two notes, and where it came from.

    Undirected, and so stored with ``a`` and ``b`` in a fixed order so
    that the same relation does not get written twice.
    """

    a: str
    b: str
    kind: str
    weight: float

    def __post_init__(self) -> None:
        if self.a > self.b:
            object.__setattr__(self, "a", self.b)
            object.__setattr__(self, "b", self.a)


def _ordered(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def words(text: str) -> set[str]:
    """Every word in a note that could carry meaning, in any script.

    No list of what to leave out. Whether a word distinguishes anything
    is a fact about the corpus and not about the word, so filtering is
    `stopwords()`'s work and happens over the whole branch.

    A set rather than a count: how often a word is repeated inside one
    note is not evidence that the note is about that word.
    """
    return {
        w for w in _WORD.findall(text.lower())
        if len(w) >= _MIN_TERM_LENGTH
    }


def terms(text: str, stop: frozenset[str] = frozenset()) -> set[str]:
    """A note's distinctive terms: its words, less the ubiquitous ones.

    ``stop`` is the corpus's, passed in rather than looked up, so that
    deriving one note's terms needs no global state and two stages that
    disagree about which words those are cannot happen by accident.
    """
    return words(text) - stop


def stopwords(
    corpus: dict[str, str],
    share: float = _SHARE,
    floor: int = _MIN_NOTES,
) -> frozenset[str]:
    """The words this corpus writes in every note, and which mean nothing.

    Counted rather than listed, so it needs no inventory of languages and
    cannot be wrong about a corpus nobody anticipated. Two notes sharing
    only these have shared grammar, not subject, and any relation drawn
    from that is the walk reaching everywhere and finding nothing.

    ``share`` is how many notes a word has to be in to stop counting, and
    ``floor`` is how many notes a corpus needs before any word is
    dropped. Both are judgements about a corpus nobody here has seen,
    which is why they are parameters: a reader who disagrees can ask for
    the other number rather than having to trust this one.
    `evaluation.ubiquity()` reports what the choice costs on a branch.
    """
    if not corpus or share > 1.0 or len(corpus) < floor:
        return frozenset()

    counts: Counter[str] = Counter()
    for text in corpus.values():
        counts.update(words(text))

    return frozenset(
        word for word, seen in counts.items() if seen >= share * len(corpus)
    )


def _targets(text: str) -> set[str]:
    """Every note this note points at, by name."""
    found = set(_WIKILINK.findall(text))
    for target in _MARKDOWN_LINK.findall(text):
        # DECISION: only relative targets name a note in the corpus. A
        # URL pointing off the device is not something we can reach.
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        found.add(target)
    return {t.strip() for t in found if t.strip()}


def _strip_ext(path: str) -> str:
    """A note path without its extension, so a link need not name one."""
    head, sep, base = path.rpartition("/")
    if "." not in base:
        return path
    return f"{head}{sep}{base.rsplit('.', 1)[0]}"


def _resolve(target: str, known: set[str], source: str) -> str | None:
    """Find the note a link names.

    Tried in order: the target as written; a relative target resolved
    against the linking note's own directory; the same with any
    extension; then a bare name against a note's stem. Lowest path wins
    at every step, so one corpus always resolves one way.
    """
    if target in known:
        return target

    # A relative markdown link is relative to the note that holds it.
    sibling = f"{source.rpartition('/')[0]}/{target}" if "/" in source else target
    for candidate in (sibling, target):
        if candidate in known:
            return candidate

    # A link need not repeat the extension, or the one the note has.
    for candidate in (sibling, target):
        matches = sorted(p for p in known if _strip_ext(p) == candidate)
        if matches:
            return matches[0]

    # A bare name matches the stem of a note, wherever it sits.
    matches = sorted(p for p in known if _strip_ext(p.rpartition("/")[2]) == target)
    return matches[0] if matches else None


def link_edges(
    notes: dict[str, str], known: set[str] | None = None
) -> tuple[list[Edge], list[tuple[str, str]]]:
    """Stage one: relations the notes state themselves.

    ``notes`` is the set to derive from, which during a refresh is only
    the notes that changed. ``known`` is what they can point at, which is
    the whole corpus: whether a link reaches anything is a fact about
    the corpus, not about the note holding it. Passing the same dict for
    both is correct when deriving from everything.

    Returns the edges and the links that pointed at nothing, which are
    reported rather than dropped: a reference to a note that is not in
    the corpus is a thing the user may want to know.
    """
    known = set(notes) if known is None else set(known)
    seen: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    dangling: list[tuple[str, str]] = []

    for source, text in notes.items():
        for target_name in sorted(_targets(text)):
            target = _resolve(target_name, known, source)
            if target is None:
                dangling.append((source, target_name))
                continue
            if target == source:
                continue
            pair = _ordered(source, target)
            if pair in seen:
                continue
            seen.add(pair)
            # A stated link is the strongest evidence there is, so it
            # outweighs anything inferred. Weight is a guess at
            # ordering only; kind is what says the same.
            edges.append(Edge(a=pair[0], b=pair[1], kind=LINK, weight=1.0))

    return edges, dangling


def cooccurrence_edges(
    notes: dict[str, str], stop: frozenset[str] = frozenset()
) -> list[Edge]:
    """Stage two: notes that talk about the same thing.

    A proxy for relation and tagged as such, so it never masquerades as
    a stated one. Compared pairwise rather than through corpus-wide term
    weights, which keeps a stage that only depends on the two notes it
    is comparing.

    ``stop`` is the corpus's ubiquitous words, from `stopwords()` over
    the whole branch, passed in rather than looked up. So the *pair* is
    still local, and a caller deriving one note's edges against the rest
    of the corpus passes the list explicitly instead of the code
    reaching into global state it does not name.
    """
    term_sets = {note_id: terms(text, stop) for note_id, text in notes.items()}
    ids = sorted(term_sets)
    edges: list[Edge] = []

    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            shared = term_sets[left] & term_sets[right]
            if len(shared) < _MIN_SHARED_TERMS:
                continue
            pair = _ordered(left, right)
            edges.append(
                Edge(
                    a=pair[0],
                    b=pair[1],
                    kind=CO_OCCURRENCE,
                    weight=float(len(shared)),
                )
            )

    return edges
