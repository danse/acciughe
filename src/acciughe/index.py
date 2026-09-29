"""Indexing a branch, and knowing when it has changed.

The behaviour here is specified by the tests in tests/test_index.py.

The graph is derived and disposable: every row in the store is
reconstructible from the notes, so the file can be deleted at any time
without loss. It lives outside the branch, because a session writes
nothing into the branch and neither does the index.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from acciughe.corpus import Entry, Unreadable, read, scan
from acciughe.graph import Graph
from acciughe.relations import LINK, cooccurrence_edges, link_edges, stopwords

# What wrote the graph, recorded so a reader whose derivation has
# changed rebuilds rather than answering from stale structure. Bump
# whenever a change in derivation would alter the rows. Bumped to 2 when
# edges entered the graph, since a store written before that held no
# relations at all and could not be read as current. Bumped to 3 when the
# stopword list was completed, since dropping "about" and the auxiliary
# verbs removes co-occurrence edges that the old derivation derived.
# Bumped to 4 when the fixed list was replaced by words counted over the
# branch, which changes the terms of every note in a corpus of any size.
CURRENT_DERIVATION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id    TEXT PRIMARY KEY,
    path  TEXT NOT NULL,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    text  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    a      TEXT NOT NULL,
    b      TEXT NOT NULL,
    kind   TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (a, b, kind)
);
CREATE INDEX IF NOT EXISTS edges_a ON edges (a);
CREATE INDEX IF NOT EXISTS edges_b ON edges (b);
"""


def _fingerprint(stop: frozenset[str]) -> str:
    """A short stable name for a set of words.

    Only ever compared to another fingerprint, so a digest is enough and
    the words themselves need not be stored. Sorted before hashing so the
    same set hashes the same however it was built.
    """
    joined = "\n".join(sorted(stop)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


class StaleGraph(Exception):
    """Raised when a caller would answer from a graph that does not match.

    The requirement is that an answer is never given from a graph that
    does not match the branch. This is the mechanism: the check is not a
    suggestion that callers may skip.
    """


@dataclass(frozen=True, slots=True)
class Read:
    """What an index or a refresh actually did.

    Reported rather than silent. The caller can see the work, and a
    refresh that changed nothing says so just as plainly.

    Named for what it is rather than for the fact that it is reported, so
    that `Report` can be the name of the one thing in this project that is
    a report: what a run of the evaluation found.
    """

    read: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unreadable: list[Unreadable] = field(default_factory=list)
    dangling: list[tuple[str, str]] = field(default_factory=list)
    rebuilt: bool = False

    @property
    def quiet(self) -> bool:
        return not (
            self.read or self.added or self.changed or self.removed
            or self.unreadable or self.dangling or self.rebuilt
        )


class Index:
    """A derived graph over a branch, and the check that keeps it true."""

    def __init__(
        self,
        branch: Path,
        store_path: Path,
        derivation: int = CURRENT_DERIVATION,
    ) -> None:
        self.branch = Path(branch)
        self.store_path = Path(store_path)
        self.derivation = derivation

    # --- store -------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.store_path)
        conn.executescript(_SCHEMA)
        return conn

    def derivation_version(self) -> int | None:
        """The derivation that wrote the stored graph, if there is one."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'derivation'"
            ).fetchone()
        return int(row[0]) if row else None

    # --- the check ---------------------------------------------------

    def _stored(self) -> dict[str, tuple[float, int]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, mtime, size FROM notes").fetchall()
        return {note_id: (mtime, size) for note_id, mtime, size in rows}

    def _present(self) -> bool:
        return self.derivation_version() is not None

    def _derivation_is_current(self) -> bool:
        return self.derivation_version() == self.derivation

    def _matches_branch(self) -> bool:
        # Deliberately holds no connection open: this is a stat-level
        # comparison and is the whole reason checking is cheap.
        stored = self._stored()
        seen = {entry.id for entry in scan(self.branch)}
        if seen != set(stored):
            return False
        for entry in scan(self.branch):
            if stored[entry.id] != (entry.mtime, entry.size):
                return False
        return True

    def current(self) -> bool:
        """Whether the graph matches the branch, and was written by us.

        False when the derivation has moved on even if nothing in the
        branch changed: stale structure is stale structure.
        """
        if not self._present() or not self._derivation_is_current():
            return False
        return self._matches_branch()

    # --- the words this corpus shares ---------------------------------

    def stopwords(self) -> frozenset[str]:
        """The words this branch writes in every note, and which say nothing.

        Read from the notes rather than stored, so a caller holding an
        index never acts on a word list the index no longer agrees with.
        `refresh` records a fingerprint rather than the words themselves,
        for the narrow job of noticing that the set moved.
        """
        return stopwords(dict(self.notes()))

    def stop_fingerprint(self) -> str:
        """Which words the graph was written with, as a name rather than a
        list.

        Recorded for the reason the derivation version is recorded: the
        words that say nothing are an input to the derivation, so a graph
        written from a different set of them is as stale as one written by
        a different version. `refresh` compares this against the branch's
        own set; it is exposed so that a reader holding a graph can see
        what wrote it rather than having to trust that the two agree.
        """
        rows = self._connect().execute(
            "SELECT value FROM meta WHERE key = 'stop'"
        ).fetchall()
        return rows[0][0] if rows else ""

    def _drift(self) -> tuple[list[Entry], list[str], list[str]]:
        """Split the branch against the store: new, changed, gone."""
        stored = self._stored()
        entries = {e.id: e for e in scan(self.branch)}

        added = [e for nid, e in sorted(entries.items()) if nid not in stored]
        changed = [
            e for nid, e in sorted(entries.items())
            if nid in stored and stored[nid] != (e.mtime, e.size)
        ]
        removed = sorted(nid for nid in stored if nid not in entries)
        return added, changed, removed

    # --- the refresh -------------------------------------------------

    def refresh(self) -> Read:
        """Bring the graph to match the branch, redoing only what changed.

        Unasked: callers are expected to call this before answering. It
        reports what it did either way.
        """
        # A reader whose derivation has changed rebuilds, even when the
        # branch is untouched: the rows would differ, so the existing
        # ones are stale structure regardless of their timestamps.
        if not self._present() or not self._derivation_is_current():
            return self._rebuild()

        added, changed, removed = self._drift()
        if not added and not changed and not removed:
            return Read()

        # Only the notes that actually changed are opened. Everything
        # else keeps the text it was indexed with.
        todo = [e for e in (*added, *changed)]
        notes, unreadable = read(self.branch, todo)

        # An unreadable note has no text and so is not a note; it is
        # reported and leaves the index if it was previously in it.
        unreadable_ids = {u.path for u in unreadable}
        with self._connect() as conn:
            for note in notes:
                conn.execute(
                    "INSERT OR REPLACE INTO notes (id, path, mtime, size, text)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (note.id, note.path, note.mtime, note.size, note.text),
                )
            for path in unreadable_ids:
                conn.execute("DELETE FROM notes WHERE id = ?", (path,))
            for note_id in removed:
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('derivation', ?)",
                (str(self.derivation),),
            )

        # Edges are recomputed for the notes that moved, and only those.
        # An edge is a function of its two endpoints, so a note that did
        # not change cannot have gained or lost a relation.
        dirty = {e.id for e in (*added, *changed)} | unreadable_ids | set(removed)

        # Which words mean nothing is a fact about the whole branch, so
        # a change in it invalidates every edge, not the one belonging to
        # the note that moved. Checked by fingerprint because the set is
        # almost always the same set: a note added to a corpus of five
        # hundred does not make a common word uncommon. Recorded, so the
        # rare case is detected rather than missed.
        stop = self.stopwords()
        if self.stop_fingerprint() != _fingerprint(stop):
            dirty |= {note_id for note_id, _text in self.notes()}
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('stop', ?)",
                (_fingerprint(stop),),
            )

        dangling = self._repopulate(dirty, unreadable_ids, stop)

        return Read(
            read=[n.path for n in notes],
            added=[e.path for e in added],
            changed=[e.path for e in changed],
            removed=removed,
            unreadable=unreadable,
            dangling=dangling,
        )

    def _repopulate(
        self,
        dirty: set[str],
        also_gone: set[str] | None = None,
        stop: frozenset[str] = frozenset(),
    ) -> list[tuple[str, str]]:
        """Re-derive edges for the given notes against the whole corpus.

        Only the dirty notes are re-derived. Every other edge was a
        function of notes that have not changed, so it stands.
        """
        texts = dict(self.notes())
        gone = dirty | (also_gone or set())

        # A link reaches only if its target is in the corpus, so a note
        # that pointed at something now gone must be re-derived even
        # though it did not itself change. Co-occurrence does not work
        # this way: it depends on the two notes and nothing else.
        pointing = self._link_sources(gone) if gone else set()
        link_dirty = (dirty | pointing) & set(texts)
        link_texts = {note_id: texts[note_id] for note_id in link_dirty}

        edges, dangling = (
            link_edges(link_texts, known=set(texts)) if link_texts else ([], [])
        )

        # Co-occurrence is pairwise, so a dirty note's edges are found by
        # pairing it with the rest of the corpus. Both stages see every
        # note, so a new note can relate to notes nobody edited.
        for edge in cooccurrence_edges(
            {k: texts[k] for k in dirty if k in texts} | texts, stop
        ):
            if edge.a in dirty or edge.b in dirty:
                edges.append(edge)

        with self._connect() as conn:
            for note_id in dirty | pointing:
                conn.execute(
                    "DELETE FROM edges WHERE a = ? OR b = ?", (note_id, note_id)
                )
            for edge in edges:
                conn.execute(
                    "INSERT OR REPLACE INTO edges (a, b, kind, weight)"
                    " VALUES (?, ?, ?, ?)",
                    (edge.a, edge.b, edge.kind, edge.weight),
                )

        return dangling

    def _link_sources(self, targets: set[str]) -> set[str]:
        """Notes that stated a link to any of the given notes."""
        if not targets:
            return set()
        marks = ",".join("?" * len(targets))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT a, b FROM edges WHERE kind = ?"
                f" AND (a IN ({marks}) OR b IN ({marks}))",
                (LINK, *targets, *targets),
            ).fetchall()
        sources: set[str] = set()
        for a, b in rows:
            if b in targets and a not in targets:
                sources.add(a)
            if a in targets and b not in targets:
                sources.add(b)
        return sources

    def _rebuild(self) -> Read:  # noqa: D401 - see refresh()
        """Build from the notes.

        Used when there is no graph, and when the derivation has changed.
        Cheap to reach for precisely because the graph holds nothing the
        notes do not.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM edges")

        entries = scan(self.branch)
        notes, unreadable = read(self.branch, entries)

        with self._connect() as conn:
            for note in notes:
                conn.execute(
                    "INSERT OR REPLACE INTO notes (id, path, mtime, size, text)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (note.id, note.path, note.mtime, note.size, note.text),
                )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('derivation', ?)",
                (str(self.derivation),),
            )

        texts = {note.id: note.text for note in notes}
        stop = stopwords(texts)
        edges, dangling = link_edges(texts)
        edges += cooccurrence_edges(texts, stop)

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('stop', ?)",
                (_fingerprint(stop),),
            )

        with self._connect() as conn:
            for edge in edges:
                conn.execute(
                    "INSERT OR REPLACE INTO edges (a, b, kind, weight)"
                    " VALUES (?, ?, ?, ?)",
                    (edge.a, edge.b, edge.kind, edge.weight),
                )

        return Read(
            read=[n.path for n in notes],
            added=[n.path for n in notes],
            unreadable=unreadable,
            dangling=dangling,
            rebuilt=True,
        )

    # --- reading -----------------------------------------------------

    def note_ids(self) -> set[str]:
        with self._connect() as conn:
            return {row[0] for row in conn.execute("SELECT id FROM notes")}

    def note_text(self, note_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        return row[0] if row else None

    def notes(self) -> list[tuple[str, str]]:
        with self._connect() as conn:
            return list(conn.execute("SELECT id, text FROM notes ORDER BY id"))

    def edges(self) -> list[tuple[str, str, str, float]]:
        with self._connect() as conn:
            return list(
                conn.execute("SELECT a, b, kind, weight FROM edges ORDER BY a, b, kind")
            )

    def edges_by_kind(self) -> dict[str, int]:
        """How much of the graph is co-occurrence rather than relation.

        The count product.md asks for, made possible by tagging every
        edge with the stage that produced it.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) FROM edges GROUP BY kind"
            ).fetchall()
        return {kind: count for kind, count in rows}

    def unconnected(self) -> set[str]:
        """Notes the graph holds that reach nothing else.

        Measured, per product.md, so an isolated note is something the
        graph can report rather than a gap in its knowledge.
        """
        connected: set[str] = set()
        for a, b, _kind, _weight in self.edges():
            connected.update((a, b))
        return self.note_ids() - connected

    def graph(self, kind: str | None = None) -> Graph:
        """The stored notes and relations, as something reachable.

        This is the seam: everything above it reaches over relations that
        were derived from a branch, rather than ones a test wrote out
        by hand.

        **A pure read of the store.** Nothing is refreshed, so what
        comes back is what the store holds, including when that has
        fallen behind the branch. Refreshing here would be convenient
        and wrong twice over: a reader that wanted to examine a graph
        could not have it replaced underneath the examination, and the
        check in ``answer_from_current_graph`` would have nothing left
        to check. Examining a graph and promising it matches the notes
        are different jobs, so they are different calls.

        ``kind`` narrows to one stage, and every note is kept either
        way. Keeping them is what makes a stage measurable: notes that
        stage left unconnected is a fact about the stage, and it is the
        comparison a third stage would have to pass. Asking for
        ``SIMILARITY`` is one of those measurements rather than a
        mistake, and answers with the corpus as it stands — which is the
        baseline that stage would be measured against.
        """
        rows = self.edges()
        if kind is not None:
            rows = [row for row in rows if row[2] == kind]
        # The store's rows are (a, b, kind, weight) and the graph's are
        # (a, b, weight), so the conversion is spelled out rather than
        # handed over: passed unchanged, the kind would land in the
        # weight slot and nothing about a reach would reveal it.
        return Graph(
            [(a, b, weight) for a, b, _kind, weight in rows],
            notes=self.note_ids(),
        )

    # --- answering ---------------------------------------------------

    def answer_from_current_graph(self) -> None:
        """Refuse to proceed unless the graph matches the branch.

        A no-op on success. Its purpose is the raise: a caller reaching
        past a stale graph gets nothing rather than a plausible answer.
        """
        if not self.current():
            raise StaleGraph(
                "the graph does not match the branch; refresh before answering"
            )
