"""Indexing a branch, and knowing when it has changed.

The behaviour here is specified by the tests in tests/test_index.py.

The graph is derived and disposable: every row in the store is
reconstructible from the notes, so the file can be deleted at any time
without loss. It lives outside the branch, because a session writes
nothing into the branch and neither does the index.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from acciughe.corpus import Entry, Unreadable, read, scan

# What wrote the graph, recorded so a reader whose derivation has
# changed rebuilds rather than answering from stale structure. Bump
# whenever a change in derivation would alter the rows.
CURRENT_DERIVATION = 1

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
"""


class StaleGraph(Exception):
    """Raised when a caller would answer from a graph that does not match.

    The requirement is that an answer is never given from a graph that
    does not match the branch. This is the mechanism: the check is not a
    suggestion that callers may skip.
    """


@dataclass(frozen=True, slots=True)
class Report:
    """What an index or a refresh actually did.

    Reported rather than silent. The caller can see the work, and a
    refresh that changed nothing says so just as plainly.
    """

    read: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unreadable: list[Unreadable] = field(default_factory=list)
    rebuilt: bool = False

    @property
    def quiet(self) -> bool:
        return not (
            self.read or self.added or self.changed or self.removed
            or self.unreadable or self.rebuilt
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

    def refresh(self) -> Report:
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
            return Report()

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

        return Report(
            read=[n.path for n in notes],
            added=[e.path for e in added],
            changed=[e.path for e in changed],
            removed=removed,
            unreadable=unreadable,
        )

    def _rebuild(self) -> Report:
        """Build from the notes.

        Used when there is no graph, and when the derivation has changed.
        Cheap to reach for precisely because the graph holds nothing the
        notes do not.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM notes")

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

        return Report(
            read=[n.path for n in notes],
            added=[n.path for n in notes],
            unreadable=unreadable,
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
