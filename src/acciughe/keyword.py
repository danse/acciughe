"""Plain keyword search: the thing to be compared against.

product.md:

  The comparison is against plain keyword search on the same branch, a
  strong baseline for notes written in your own words.

Two choices carry the weight, and both were argued against something
easier.

**It is a real full-text engine.** FTS5 with BM25 ranking and Porter
stemming, not a loop over a list of words. The spec asks for a *strong*
baseline, and the reason it asks is that a weak one would make the
graph look good for the wrong reason. A reader who finds this
unsatisfactory has a search box and can see exactly what it matched.

**It reads the branch, never the graph.** A baseline built on the index
would inherit the index's mistakes, so a question the graph gets wrong
would be a question the baseline also gets wrong, and the comparison
would be blind to the failure it exists to catch. Here the two
disagree, which is the point.

The index it builds is disposable and lives in memory. Nothing here is
derived from anything: it is a view of the notes, and a view can be
thrown away and rebuilt without loss.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from acciughe.corpus import Unreadable, read, scan
from acciughe.relations import stopwords, words

# Letters and digits, in any script. A corpus in more than one language
# would not be searched by an ASCII pattern, and the notes are not
# required to be English.
_WORD = re.compile(r"[^\W_]+")

_SCHEMA = """
CREATE VIRTUAL TABLE notes USING fts5(
    path UNINDEXED,
    body,
    tokenize = 'porter unicode61'
);
"""


class KeywordSearch:
    """Keyword search over the notes on a branch."""

    def __init__(self, branch: Path) -> None:
        self.branch = Path(branch)
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(_SCHEMA)
        self._stamped: dict[str, tuple[float, int]] = {}
        self._unreadable: list[Unreadable] = []
        self._stop: frozenset[str] | None = None
        self._sync()

    # --- keeping up with the branch ----------------------------------

    def _sync(self) -> None:
        """Reindex only if the branch has actually moved.

        The check is stat-only and the rebuild is total, which is cheap
        because a corpus of notes is small and because the caller was
        never going to remember to ask.
        """
        entries = scan(self.branch)
        stamped = {e.id: (e.mtime, e.size) for e in entries}
        if stamped == self._stamped:
            return

        notes, unreadable = read(self.branch, entries)
        with self._conn:
            self._conn.execute("DELETE FROM notes")
            self._conn.executemany(
                "INSERT INTO notes (path, body) VALUES (?, ?)",
                [(n.path, n.text) for n in notes],
            )
        self._stamped = stamped
        self._unreadable = unreadable
        # The ubiquitous words are a fact about these notes, so they
        # change exactly when the notes do. Recomputing them per query
        # would re-tokenise the whole branch for every question in a
        # run, and a run is a long list of questions.
        self._stop = None

    @property
    def unreadable(self) -> list[Unreadable]:
        """Files on the branch that are not notes, reported not dropped.

        Read fresh rather than cached, so it is true of the branch as it
        stands rather than as it stood when the last query ran.
        """
        self._sync()
        return list(self._unreadable)

    # --- searching ---------------------------------------------------

    def search(self, query: str, limit: int | None = None) -> list[str]:
        """The paths of the notes a reader would find, best match first.

        Any term is enough: the words someone types at their own notes
        are alternatives, and requiring all of them would fail queries
        for reasons the reader cannot see. Those failures would then be
        read as the graph being useful, which is the outcome the
        comparison exists to prevent.
        """
        self._sync()
        terms = self._terms(query, self.stopwords())
        if not terms:
            return []

        match = " OR ".join(f'"{term}"' for term in terms)
        sql = "SELECT path FROM notes WHERE notes MATCH ? ORDER BY bm25(notes)"
        if limit is not None:
            sql += " LIMIT ?"
        params = (match, limit) if limit is not None else (match,)
        return [row[0] for row in self._conn.execute(sql, params)]

    def stopwords(self) -> frozenset[str]:
        """The words this branch writes in every note.

        Counted here rather than imported from the graph, because a
        baseline that read the index's list would inherit the index's
        mistakes. The words are the same and the reasoning is not shared,
        which is the arrangement that lets the two disagree about a
        question — the point of the comparison.

        Counted once per state of the branch rather than once per query:
        the count cannot change between two queries, and a run asks a
        long list of them.
        """
        if self._stop is not None:
            return self._stop
        self._stop = stopwords({
            row[0]: row[1]
            for row in self._conn.execute("SELECT path, body FROM notes")
        })
        return self._stop

    @staticmethod
    def _terms(query: str, stop: frozenset[str]) -> list[str]:
        """The words of a query that carry any meaning, in order, unrepeated.

        A query of nothing but ubiquitous words returns nothing rather
        than everything, or "what is it" would return the corpus.

        The same tokenizer as the graph's, so a word dropped here is a
        word dropped there; which words are dropped is this corpus's own
        answer, not a list that predates the notes.
        """
        keep = words(query) - stop
        seen: dict[str, None] = {}
        for word in _WORD.findall(query.lower()):
            if word in keep:
                seen.setdefault(word, None)
        return list(seen)
