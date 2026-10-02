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
from functools import cache
from math import log

from stopwordsiso import has_lang, langs
from stopwordsiso import stopwords as stopwordsiso

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

# DECISION: a language counts as written in this corpus when its stopword
# list accounts for at least this fraction of the best one. A ratio and not
# a share of the corpus's words, because the two are measured against
# different things: a fixed fraction of the vocabulary would be a threshold
# about how much grammar a corpus has, and would drop a language written
# briefly in a corpus dominated by another.
#
# Half, and the gap it sits in is wide enough not to need care: measured
# over the corpus's whole vocabulary, an English branch scores a quarter
# against English and about a twentieth against everything else, and a
# mixed Italian-and-English one scores a fifth each way. Half of the best
# is between those and far above the noise.
#
# The direction to be wrong in is the safe one. A language missed here
# leaves its function words in the graph as terms, which costs edges and
# puts the words back where the count used to; a language wrongly included
# drops a word that was doing something, which is the mistake that loses a
# relation.
_DETECTION_RATIO = 2

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

    No list of what to leave out. Whether a word says anything is a fact
    about the language a note is written in, so filtering is
    `stopwords()`'s work and happens over the whole branch, which is
    what says which languages are in play.

    A set rather than a count: how often a word is repeated inside one
    note is not evidence that the note is about that word.
    """
    return {
        w for w in _WORD.findall(text.lower())
        if len(w) >= _MIN_TERM_LENGTH
    }


def distinguishing(notes: dict[str, str], stop: frozenset[str]) -> dict[str, float]:
    """How much each term tells one note from another, in this corpus.

    **A term's weight is the log of how few notes hold it.** A term in
    every note is worth nothing — it cannot separate anything, so a match
    on it is a coincidence of vocabulary rather than of subject. A term in
    two notes is worth almost the most there is, because it is close to
    naming them.

    DECISION: log(N / notes holding the term), replacing a plain count of
    shared terms, and this reverses an earlier decision that ranked seeds
    by how many terms they shared. The count was argued on a monolingual
    corpus where a function word is in most notes and so shared by almost
    every candidate at once. On a corpus in two languages that argument
    fails: an Italian corpus's function words are in every Italian note
    and none of the English ones, so the most widespread word in a
    131-note Italian-and-English branch reached 38% — below every
    threshold `stopwords()` could have set, which therefore found an
    empty set and ranked the seeds of "why did the press jam?" by how many
    notes contained the words *the*, *did* and *why*. Thirty-six seeds came
    back and the first was a note about a person's history. Language lists
    closed that particular hole — *the* and *did* are in the English list
    whatever the corpus looks like — and the weighting stayed, because a
    word in every note still weighs nothing and the lists cannot know
    which of the reader's own words those are.

    A weight rather than a threshold, because no cutoff is discoverable
    here: the distribution of how widespread each term is decays smoothly
    with no cliff in it, so a share set to exclude a habit of writing also
    excludes subject words by degrees and there is no value that is right
    rather than roughly defensible. Dividing by the term's own frequency
    instead makes the corpus supply the weights and cannot come back empty.

    A term in every note scores zero rather than being dropped, because
    `seeds_for` still treats a zero-weight match as a match: it is a weak
    one, and a question that names a note only by words everybody writes
    is about that note more than nothing at all.
    """
    held: Counter[str] = Counter()
    for note_id, text in notes.items():
        held.update(terms(text, stop) | {_name(note_id)})

    total = len(notes)
    if not total:
        return {}
    return {term: log(total / seen) for term, seen in held.items()}


def _name(note_id: str) -> str:
    """A note's own name, as the term a question would use to mean it.

    A note's name is a term like any other, because a question that
    mentions a note's name is about that note: `tracsis/acquisto/halving.hs`
    and "halving" are the same reference, and a match that could not see
    that would seed on whatever else the note happened to share.
    """
    base = note_id.rpartition("/")[2]
    return base.rsplit(".", 1)[0].lower() if "." in base else base.lower()



def terms(text: str, stop: frozenset[str] = frozenset()) -> set[str]:
    """A note's distinctive terms: its words, less what its language says
    nothing with.

    ``stop`` is the corpus's, passed in rather than looked up, so that
    deriving one note's terms needs no global state and two stages that
    disagree about which words those are cannot happen by accident.
    """
    return words(text) - stop


@cache
def _stopword_lists() -> dict[str, frozenset[str]]:
    """Every language's stopwords, read once.

    Fifty-eight lists, and the corpus is read once per language otherwise;
    a refresh of a real branch would rather read them once.
    """
    return {
        lang: frozenset(stopwordsiso(lang))
        for lang in langs()
        if has_lang(lang)
    }


def _detect_languages(corpus: dict[str, str]) -> frozenset[str]:
    """Which languages these notes are written in.

    Scored by how much of the corpus each language's list accounts for,
    which is the one measurement that says a language rather than guessing
    it: an English corpus covers a quarter of its words with the English
    list and about a twentieth with any other, so the languages that clear
    half of the best score are the ones actually being written.

    A stopword list as the detector rather than a keyword list beside it.
    A keyword list says what marks a language, and every one of them is a
    word on somebody's own stopword list — `di` is Italian and also the
    French past tense of `dire`, `die` is German and also an English verb.
    Asking which list best accounts for the words is the same question
    without the ambiguity, and it needs no list kept in step with the
    other one.

    Both halves of a mixed corpus are found, because the union of two
    languages is still best explained by each of them.

    Nothing is detected in a corpus whose words no list accounts for — a
    branch of nouns, or one word — and that is the honest answer rather
    than a default: a guess would strip words from a corpus in a language
    nobody thought to enumerate.
    """
    vocabulary = words(" ".join(corpus.values()))
    if not vocabulary:
        return frozenset()

    scores = {
        lang: len(vocabulary & every) / len(vocabulary)
        for lang, every in _stopword_lists().items()
    }
    best = max(scores.values())
    if not best:
        return frozenset()
    return frozenset(
        lang for lang, score in scores.items() if score >= best / _DETECTION_RATIO
    )


def stopwords(corpus: dict[str, str]) -> frozenset[str]:
    """The words these notes are written with and mean nothing by.

    The languages the corpus is written in are detected and their lists
    unioned, so an Italian corpus drops `della` and an English one drops
    `the`, and a corpus in both drops both.

    Replaces a count over the corpus, which asked what share of the notes
    write a word and needed a threshold to answer it. That threshold was
    undecidable — measured on a 131-note Italian-and-English branch no word
    reached half the notes, so the count found nothing to drop and the
    function words went into the graph as terms — and it needed a floor as
    well, because below three notes a count cannot tell a function word
    from the one subject the notes agree on. A list is a fact about a
    language, so it needs neither number, and `agenda.md` no longer has a
    share to settle.

    Length-filtered to `_MIN_TERM_LENGTH` because a one or two letter word
    is not a term by `words()`'s own rule, and a stopword set that
    disagreed with it would make `terms()` depend on which set it was
    given.
    """
    found: set[str] = set()
    for lang in _detect_languages(corpus):
        found |= _stopword_lists()[lang]
    return frozenset(
        word for word in found if len(word) >= _MIN_TERM_LENGTH
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

    ``stop`` is the corpus's words that say nothing, from `stopwords()`
    over the whole branch, passed in rather than looked up. So the *pair*
    is still local, and a caller deriving one note's edges against the
    rest of the corpus passes the list explicitly instead of the code
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
