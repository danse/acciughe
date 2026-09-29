"""Reaching along relations, and showing the work.

The behaviour here is specified by the tests in tests/test_graph.py.

Relations are undirected. A relation between two notes does not, in
this graph, have a direction; anything that does would need the spec to
say so, and it does not.

**The walk is queries.** The relations are held in a table, and a reach
expands one depth at a time: the notes found at the last depth are the
frontier, and one indexed join on those names is the whole of the next
step. Every note is expanded once in a walk, so the traversal costs what
the relations cost and not their square.

A reach costs one thing more, and it is the price of showing the work: a
path is held for every note reached, so a reach costs what those paths
cost. On a well-connected corpus that is nothing next to the relations.
On a corpus that is one long chain they are the same order, since every
note is as far from the seed as the corpus is long. That is what
``paths`` is for, and it was the same before the relations were held in a
table.

What this buys and what it does not, said plainly:

- The relations are still read into the graph before anything walks them,
  because a graph is a snapshot. A graph that queried the store on every
  step would stop being a snapshot, and ``Index.graph`` promises one — a
  reader examining a graph must not have it replaced underneath the
  examination. So this is not about reading fewer edges, and nothing here
  should be sold as making a large corpus cheap.
- It was one recursive CTE, and measuring said it was worse than the
  dictionary it replaced. ``UNION`` dedups the whole row, a row held a
  depth, and a note that can be arrived at at one depth can be arrived at
  at the next by walking a relation and coming back — so the query held
  every note at every depth, and a 2000-note graph took minutes. A
  recursive CTE cannot be told what it has already seen, so the
  already-seen is kept out here and the query is asked one depth at a
  time. That is the reason, and it is worth more than the tidiness of
  having been one statement.
- ``max_depth`` is a count of steps, not a clause in a statement, and an
  unbounded walk needs no bound of its own because the frontier empties.
- The path to a note is read back off the walk, not derived afterwards.
  A note one relation out of two frontier notes has to be settled at the
  step where both are in hand, and asking the graph again for each step of
  each path would walk it a second time — on a corpus that is one long
  chain, once per step of the path to every note.
- ``cycles`` is still a walk in Python. A cycle is not a property one
  query returns the way a reach is — the loop back through each note is a
  different question each time round, and the claim that it could be one
  pass was wrong. It builds its adjacency on demand, and only when asked.
- Which shortest path is reported is a choice, and it is made by order:
  shortest first, and ties broken by name. A name is the only tie-break
  available that is the same however the relations were stored or
  returned, so an answer over the same corpus is the same answer twice.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

# An edge with no stated weight is as strong as any other.
_DEFAULT_WEIGHT = 1.0

_SCHEMA = """
CREATE TABLE edges (
    a      TEXT NOT NULL,
    b      TEXT NOT NULL,
    weight REAL NOT NULL
);
CREATE INDEX edges_a ON edges (a);
CREATE INDEX edges_b ON edges (b);

-- Relations have no direction, so walking them needs both. A self-edge
-- is excluded because a note cannot reach itself: keeping it would put
-- the note at a relation's distance as well as at distance zero, and the
-- walk would report one note at two depths for no reason a reader could
-- see. It is still held as a relation; it just leads nowhere.
CREATE VIEW adjacency AS
    SELECT a AS note, b AS other, weight FROM edges WHERE a <> b
    UNION ALL
    SELECT b AS note, a AS other, weight FROM edges WHERE a <> b;
"""


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


def _by_depth_then_note(depths: dict[str, int]):
    """Report shortest first, and break a tie by name.

    Two paths of the same length are both correct answers, and which one
    is reported should not depend on the order the store happened to hand
    its rows over in. A name is the only ordering available that is the
    same on every run.
    """
    return lambda note: (depths[note], note)


class Graph:
    """An undirected set of weighted relations between notes.

    The relations are held in a table rather than in a dictionary, so that
    expanding them is a query and there is one traversal to keep correct.
    Everything else about a graph is still a value: it is built, it is
    examined, and it does not follow the store it came from.
    """

    def __init__(self, edges, notes=()) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(_SCHEMA)
        self._notes: set[str] = set(notes)
        self._adjacency_cache: dict[str, list[tuple[str, float]]] | None = None

        rows = []
        for edge in edges:
            a, b, *rest = edge
            weight = rest[0] if rest else _DEFAULT_WEIGHT
            self._notes.update((a, b))
            # A relation has no direction, so which end is called ``a`` is
            # a naming convention and not a fact. It is settled here, once,
            # so every read of the table can rely on it: the ends are given
            # either way round by a caller and in one order by the store.
            rows.append((min(a, b), max(a, b), float(weight)))
        self._conn.executemany("INSERT INTO edges VALUES (?, ?, ?)", rows)

    @property
    def _adjacency(self) -> dict[str, list[tuple[str, float]]]:
        """The relations as a dictionary, built on demand.

        Only `cycles` wants this, so paying for it is left to the caller
        that asked rather than charged to every graph.
        """
        if self._adjacency_cache is None:
            built: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for note, other, weight in self._conn.execute(
                "SELECT note, other, weight FROM adjacency"
            ):
                built[note].append((other, weight))
            self._adjacency_cache = built
        return self._adjacency_cache

    def unconnected(self) -> set[str]:
        """Notes the graph holds that reach nothing else.

        A measured property: a large isolated set is a graph that is not
        doing its job, so it has to be countable.
        """
        connected = {
            row[0] for row in self._conn.execute("SELECT DISTINCT note FROM adjacency")
        }
        return {note for note in self._notes if note not in connected}

    def notes(self) -> set[str]:
        """Every note the graph knows about, reached or not.

        Includes notes given that are connected to nothing: the count of
        unconnected notes is a measured property, not an absence.
        """
        return set(self._notes)

    def edges(self) -> list[tuple[str, str, float]]:
        """The relations held, each with its ends in a fixed order.

        Read off the same table a reach walks, so it cannot disagree with
        one. Each relation is held once, with its ends in a fixed order,
        because reporting it both ways round would double every count made
        of it.
        """
        return [
            (a, b, weight)
            for a, b, weight in self._conn.execute(
                "SELECT a, b, weight FROM edges ORDER BY a, b"
            )
        ]

    def reach(self, seeds, max_depth: int | None = None) -> Reach:
        """Walk relations out from the seeds.

        Depth is counted in relations traversed, and each note is
        reported by a shortest path to it. ``left_out`` is everything in
        the graph the walk did not reach, so a bounded examination states
        its own limits.
        """
        seeds = list(seeds)
        for seed in seeds:
            self._notes.add(seed)
        if not seeds:
            return Reach(left_out=set(self._notes))

        depths, led = self._walk(seeds, max_depth)

        return Reach(
            reached=set(depths),
            paths=self._trails(depths, led),
            _depths=depths,
            truncated=self._truncated_at(depths, max_depth),
            left_out=self._notes - set(depths),
        )

    def _walk(
        self, seeds: list[str], max_depth: int | None
    ) -> tuple[dict[str, int], dict[str, str]]:
        """The walk: one query per relation of depth, out from the frontier.

        This was one recursive CTE, and it was wrong, and measuring is what
        showed it. ``UNION`` dedups the whole row, so a row of
        ``(note, depth)`` is held once per depth a note can be arrived at —
        and a note that can be arrived at at depth *d* can also be arrived
        at at depth *d + 1*, by walking a relation and coming back. So
        every note was emitted at every depth up to the bound, and a walk
        with no bound is bounded by the graph: over a connected corpus that
        is a row per note per note, and a 2000-note graph took minutes
        where a dictionary walk takes milliseconds.

        The fix is that the CTE has no way to be told which notes it has
        already been, so the already-seen is kept outside the query and the
        query is asked one depth at a time. Each step expands only the
        frontier — the notes found at the last depth — so every note is
        expanded once in the whole walk and the total is proportional to
        the relations rather than to their square.

        The cost of that is a loop, which is what this replaced, and the
        loop is over depths rather than over notes: a bounded walk at depth
        two is two queries, not one per note. An unbounded walk needs no
        bound of its own, because the frontier empties.

        Returns how far each note is and, for each one that was not a seed,
        the note it was reached from — so every path can be read back off
        the walk instead of being looked up again.
        """
        reached: dict[str, int] = dict.fromkeys(seeds, 0)
        led: dict[str, str] = {}
        frontier = list(reached)
        depth = 0
        while frontier and (max_depth is None or depth < max_depth):
            depth += 1
            step = self._step(frontier, reached)
            led.update(step)
            reached.update(dict.fromkeys(step, depth))
            frontier = list(step)
        return reached, led

    def _step(self, frontier: list[str], reached: dict[str, int]) -> dict[str, str]:
        """The notes one relation out from the frontier, and what led to each.

        The frontier goes in as a list of names and the notes come back in
        one order, so a walk over the same graph is the same walk however
        the relations were stored.

        Which frontier note led to a note is decided here rather than
        later, because a note can be one relation out of two of them at
        once and this is the only moment both are in hand. It is recorded
        as it is read rather than looked up afterwards, so building the
        path to a note costs no reads of its own: asking the graph again
        for each step of each path would make a walk cost what the paths
        cost, and on a corpus that is one long chain the paths are as long
        as the corpus.
        """
        placeholders = ",".join(["?"] * len(frontier))
        via: dict[str, str] = {}
        for other, source in self._conn.execute(
            f"SELECT other, note FROM adjacency WHERE note IN ({placeholders})",
            frontier,
        ):
            if other in reached:
                continue
            held = via.get(other)
            if held is None or source < held:
                via[other] = source
        return via

    def _trails(
        self, depths: dict[str, int], led: dict[str, str]
    ) -> list[list[str]]:
        """A path to every note the walk reached, shortest first.

        Read back off the walk rather than derived from the graph, because
        the walk already had to decide which note led to which — a note
        one relation out of two at once has to be settled at the step
        where both are in hand — and asking the graph again for every step
        of every path would be the same walk done twice.

        Which of two shortest paths is shown was decided by name when the
        walk read the relations, so two shortest paths are both correct,
        one is reported, and it is the same one however the relations were
        stored.
        """
        found = []
        for note in sorted(depths, key=_by_depth_then_note(depths)):
            trail = [note]
            while note in led:
                note = led[note]
                trail.append(note)
            found.append(trail[::-1])
        return found

    def _truncated_at(self, depths: dict[str, int], max_depth: int | None) -> bool:
        """Whether the bound stopped the walk short of something.

        Truncated means the bound cut a walk off, which is not the same as
        the walk having stopped. A bound that lands on the end of the graph
        reached everything there was: there is a note at the bound, and
        behind it is nothing new, so the examination is whole and says so.

        The tempting test — is there a note at the bound, and does it have
        any relation — is wrong, because the relation back to where the
        walk came from is a relation. It reports a walk that reached
        everything as cut short, and says so beside a ``left_out`` that is
        empty: two claims about one examination that cannot both be true.
        So what is asked is whether anything behind the bound was *not*
        reached, which is the question ``left_out`` answers and the only
        one where the two agree.
        """
        if max_depth is None:
            return False
        at_bound = [note for note, depth in depths.items() if depth == max_depth]
        if not at_bound:
            return False
        at = ",".join(["?"] * len(at_bound))
        already = ",".join(["?"] * len(depths))
        return (
            self._conn.execute(
                f"""SELECT 1 FROM adjacency
                    WHERE note IN ({at}) AND other NOT IN ({already})
                    LIMIT 1""",
                [*at_bound, *depths],
            ).fetchone()
            is not None
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
