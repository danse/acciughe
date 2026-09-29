"""Reaching along relations, and showing the work.

The behaviour here is specified by the tests in tests/test_graph.py.

Relations are undirected. A relation between two notes does not, in
this graph, have a direction; anything that does would need the spec to
say so, and it does not.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field

# An edge with no stated weight is as strong as any other.
_DEFAULT_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class Cycle:
    """A loop back: the notes on it, in order, closing on the first."""

    notes: list[str]

    @property
    def length(self) -> int:
        return len(self.notes) - 1  # the closing repeat is not a step


@dataclass(frozen=True, slots=True)
class Reach:
    """What a reach found, and how it got there.

    A reach both answers and shows its work: ``paths`` is retained so the
    structure behind an answer can be examined, and ``left_out`` records
    what the examination did not cover.
    """

    reached: set[str] = field(default_factory=set)
    paths: list[list[str]] = field(default_factory=list)
    _depths: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    left_out: set[str] = field(default_factory=set)

    def depth_of(self, note_id: str) -> int:
        return self._depths[note_id]


class Graph:
    """An undirected set of weighted relations between notes."""

    def __init__(self, edges, notes=()) -> None:
        self._adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._notes: set[str] = set(notes)
        for edge in edges:
            a, b, *rest = edge
            weight = rest[0] if rest else _DEFAULT_WEIGHT
            self._connect(a, b, weight)

    def _connect(self, a: str, b: str, weight: float) -> None:
        self._notes.update((a, b))
        if a == b:
            return  # a self-relation reaches nothing new
        self._adjacency[a].append((b, weight))
        self._adjacency[b].append((a, weight))

    def unconnected(self) -> set[str]:
        """Notes the graph holds that reach nothing else.

        A measured property: a large isolated set is a graph that is not
        doing its job, so it has to be countable.
        """
        return {note for note in self._notes if not self._adjacency.get(note)}

    def notes(self) -> set[str]:
        """Every note the graph knows about, reached or not.

        Includes notes given that are connected to nothing: the count of
        unconnected notes is a measured property, not an absence.
        """
        return set(self._notes)

    def reach(self, seeds, max_depth: int | None = None) -> Reach:
        """Walk relations out from the seeds.

        Depth is counted in relations traversed, and the walk is
        breadth-first so that a note is always reported by a shortest
        path. ``left_out`` is everything in the graph the walk did not
        reach, so a bounded examination states its own limits.
        """
        seeds = list(seeds)
        for seed in seeds:
            self._notes.add(seed)

        reached: set[str] = set()
        depths: dict[str, int] = {}
        paths: list[list[str]] = []
        truncated = False

        queue: deque[tuple[str, int, list[str]]] = deque(
            (seed, 0, [seed]) for seed in seeds
        )
        for seed in seeds:
            reached.add(seed)
            depths[seed] = 0
            paths.append([seed])

        while queue:
            note_id, depth, trail = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                # There may be more behind these notes; say so rather
                # than presenting a bounded walk as a complete one.
                if self._adjacency.get(note_id):
                    truncated = True
                continue
            for neighbour, _weight in self._adjacency.get(note_id, ()):
                if neighbour in reached:
                    continue
                reached.add(neighbour)
                depths[neighbour] = depth + 1
                step = [*trail, neighbour]
                paths.append(step)
                queue.append((neighbour, depth + 1, step))

        return Reach(
            reached=reached,
            paths=paths,
            _depths=depths,
            truncated=truncated,
            left_out=self._notes - reached,
        )

    def cycles(self) -> list[Cycle]:
        """Find what loops back.

        Iterative depth-first search with an explicit stack: a large
        corpus is exactly where recursion would run out of stack, and a
        cycle-heavy graph is exactly where it would run out first.
        """
        found: list[Cycle] = []
        seen_signatures: set[tuple[str, ...]] = set()

        for start in sorted(self._notes):
            if not self._adjacency.get(start):
                continue
            # trail maps a note to how it was first entered; seeing a node
            # already on the trail is what closes a loop.
            trail: dict[str, str | None] = {start: None}
            order: list[str] = [start]
            stack: list[tuple[str, int]] = [(start, 0)]

            while stack:
                node, index = stack[-1]
                neighbours = self._adjacency.get(node, ())
                if index >= len(neighbours):
                    stack.pop()
                    order.pop() if order else None
                    continue
                stack[-1] = (node, index + 1)
                neighbour = neighbours[index][0]
                # In an undirected graph, walking an edge and returning
                # along it is not a loop — it is the same edge twice. A
                # real cycle needs at least three notes.
                if neighbour == start and len(order) >= 3:
                    self._record(found, seen_signatures, order)
                elif neighbour not in trail:
                    trail[neighbour] = node
                    order.append(neighbour)
                    stack.append((neighbour, 0))

        return found

    @staticmethod
    def _record(found: list[Cycle], seen: set[tuple[str, ...]], order: list[str]) -> None:
        # The same loop is discovered once per note on it; keep the first.
        rotated = [order[i] for i in range(len(order))]
        pivot = min(range(len(rotated)), key=lambda i: rotated[i])
        canonical = tuple(rotated[pivot:] + rotated[:pivot])
        if canonical in seen:
            return
        seen.add(canonical)
        found.append(Cycle(notes=[*canonical, canonical[0]]))
