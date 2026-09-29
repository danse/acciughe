"""A corpus, and what a note is.

The behaviour here is specified by the tests in tests/test_corpus.py.
Where this module makes a choice the spec does not make, the test that
pins it is marked DECISION.
"""

from __future__ import annotations

import os
import stat as stat_mod
from dataclasses import dataclass
from pathlib import Path

# A note is text. Most binary content decodes without error under a
# permissive codec, so decoding is checked strictly and NUL is treated
# as the giveaway that the bytes were never text to begin with.
_ENCODING = "utf-8"


@dataclass(frozen=True, slots=True)
class Entry:
    """A file seen by the walk, identified without reading it.

    Everything here comes from ``stat``. This is what makes the recheck
    cheap: the branch is walked and compared, and nothing is opened.
    """

    id: str
    path: str  # POSIX, relative to the branch root
    mtime: float
    size: int


@dataclass(frozen=True, slots=True)
class Note:
    """An entry that decoded as text."""

    id: str
    path: str
    mtime: float
    size: int
    text: str


@dataclass(frozen=True, slots=True)
class Unreadable:
    """A file the branch holds that is not a note.

    Carried rather than dropped: product.md requires that a file which
    cannot be read as text is reported when the branch is indexed, so
    the walk must be able to say what it left out.
    """

    path: str
    reason: str
    relative_to_branch: bool = True


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def scan(branch: Path) -> list[Entry]:
    """Walk the branch and return what is there, without reading it.

    Hidden entries are excluded along with their contents, and symlinked
    directories are not followed. A set of already-visited real
    directories is kept so that a loop cannot make the walk unbounded.
    """
    root = Path(branch).resolve()
    seen_dirs: set[str] = set()
    entries: list[Entry] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)

        # Drop hidden subtrees and prune the walk into them. Skipping
        # them in dirnames without pruning would still descend.
        dirnames[:] = [d for d in dirnames if not _is_hidden(d)]

        try:
            marker = str(here.resolve())
        except OSError:
            marker = str(here)

        if marker in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(marker)

        for name in filenames:
            if _is_hidden(name):
                continue
            full = here / name
            try:
                st = full.stat()  # follows symlinks: a linked file is read
            except OSError:
                continue
            if not _is_regular(st):
                continue
            rel = _relative(full, root)
            entries.append(
                Entry(
                    id=rel,
                    path=rel,
                    mtime=st.st_mtime,
                    size=st.st_size,
                )
            )

    entries.sort(key=lambda e: e.path)
    return entries


def _is_regular(st: os.stat_result) -> bool:
    """A device, socket, or fifo is not a note."""
    return stat_mod.S_ISREG(st.st_mode)


def _decode(raw: bytes) -> str:
    """Decode strictly, and refuse bytes that were never text."""
    text = raw.decode(_ENCODING)
    if "\x00" in text:
        raise ValueError("contains NUL")
    return text


def read(branch: Path, entries: list[Entry]) -> tuple[list[Note], list[Unreadable]]:
    """Read the given entries, partitioning into notes and reports."""
    root = Path(branch).resolve()
    notes: list[Note] = []
    unreadable: list[Unreadable] = []

    for entry in entries:
        full = root / entry.path
        try:
            raw = full.read_bytes()
        except OSError as exc:
            unreadable.append(Unreadable(entry.path, f"could not read: {exc}"))
            continue
        try:
            text = _decode(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            unreadable.append(
                Unreadable(entry.path, f"not text: {exc}")
            )
            continue
        notes.append(
            Note(
                id=entry.id,
                path=entry.path,
                mtime=entry.mtime,
                size=entry.size,
                text=text,
            )
        )

    return notes, unreadable
