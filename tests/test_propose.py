"""Questions the graph proposes, taken from the corpus rather than written.

The gap this file closes. Everything measured so far was measured on a
question someone wrote by hand, which is the one kind of question this
product is least likely to be asked. product.md is explicit about where
the set comes from:

  An agent walking the graph proposes the questions, so they come from
  the corpus rather than from memory.

  This finds failures of one kind, since a question is asked only if the
  graph noticed something.

  The graph proposes; the notes decide: a question counts only when the
  notes support it, or a structure the reader invented becomes its own
  proof.

The last one is why this file is mostly about what a proposal is *not*
allowed to claim. Nothing here has read the notes, so nothing here may
decide a question is good; the whole of that is `question_verdict`, and
the tests here end where that one begins.
"""

import pytest

from collections import Counter

from acciughe.evaluation import KEPT, question_verdict
from acciughe.index import Index
from acciughe.keyword import KeywordSearch
from acciughe.propose import (
    DEFAULT_DEPTH,
    DEFAULT_PER_NOTE,
    Proposal,
    proposals,
    questions_in,
)

# --- What counts as a question ------------------------------------------

def test_a_sentence_that_asks_something_is_a_question():
    assert questions_in("Why did the border jam the printer?") == [
        "Why did the border jam the printer?"
    ]


def test_a_statement_is_not_a_question():
    assert questions_in("The printer jams on anything with a border.") == []


def test_a_heading_that_asks_is_a_question_without_its_hash():
    """A user asking the question would not type the hash, and a question
    with a hash in it is one nobody wrote."""
    assert questions_in("## Why does the queue clear so slowly?") == [
        "Why does the queue clear so slowly?"
    ]


def test_a_heading_that_names_a_subject_is_not_a_question():
    """An "Open questions" list heading is not a question the user asked.
    Proposing it would put a question in the set that has no asker."""
    assert questions_in("## Open questions\n- one\n- two") == []


def test_a_list_item_that_asks_is_a_question_without_its_bullet():
    assert questions_in("- Does the same apply upstairs?") == [
        "Does the same apply upstairs?"
    ]


def test_a_bullet_that_follows_a_sentence_is_not_part_of_the_question():
    """The bullet is at the start of a line the corpus wrote, so stripping
    it there was not enough — a bullet after a sentence end is at the
    start of nothing the split saw, and it came through with the
    question. `- quanti giorni saranno passati ?` is a question with a
    list marker in it, and the marker is the part nobody asked."""
    assert questions_in("Ragusa is in Sicily. - the electric bike there?") == [
        "the electric bike there?"
    ]


def test_a_comment_marker_is_not_part_of_the_question_either():
    """`--` opens a line comment, and a corpus may hold code as well as
    prose. A run of markers is one marker, stripped for the same reason
    the bullet is: it is not part of what was asked. The pipe in
    Haddock's `-- |` is left alone on purpose — it is also a guard in a
    pattern match, and no proposal of a code file is worth that."""
    assert questions_in("-- Any fusion?") == ["Any fusion?"]


def test_a_minus_before_a_number_is_not_a_marker():
    """A pattern loose enough to strip `--` is loose enough to strip the
    sign off `-5`, and a question is lifted rather than rewritten. This
    is the case that keeps the pattern's lookahead there."""
    assert questions_in("-5 and what about the rest?") == [
        "-5 and what about the rest?"
    ]


def test_a_question_mark_on_its_own_is_not_a_question():
    """A question mark is what makes a span a question, not what makes it
    one. A span with nothing in it but punctuation asks nothing, and
    without this a note that once held a question there proposed `??` and
    spent a turn of the evaluation on it."""
    assert questions_in("and then what? ?") == ["and then what?"]
    assert questions_in("??") == []


def test_only_the_asking_part_of_a_line_is_the_question():
    """Splitting on sentence ends is what keeps a link and a statement
    out of the question. Proposed verbatim, "See [[queue]] and the
    printer jam?" is a question nobody asked."""
    assert questions_in("See [[queue]] and the printer jams. Why?") == ["Why?"]


def test_a_question_keeps_its_own_punctuation_and_words():
    """It is lifted, not rephrased. product.md wants the question to come
    from the corpus, and rephrasing is the reader's memory arriving
    through the back door."""
    text = "Didn't the second header break the DE-9?"

    assert questions_in(text) == [text]


def test_questions_are_returned_in_the_order_the_note_asks_them():
    text = "Why now? And why not then? What changed?"

    assert questions_in(text) == ["Why now?", "And why not then?", "What changed?"]


def test_two_questions_in_one_note_are_two_questions():
    assert len(questions_in("Why now? What changed?")) == 2


def test_a_note_with_no_question_offers_nothing():
    assert questions_in("The store is derived and can be deleted.") == []


# --- Walking out from a question ----------------------------------------

@pytest.fixture
def branch(tmp_path):
    """A corpus shaped like the one product.md describes: a question the
    user wrote down, and answers written at a distance from it in words
    that are not its own."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "queue.md").write_text(
        "## Why does clearing a job take so long?\n"
        "The spooler restarts between jobs.\n"
        "See [[engineering-hall]] and [[duplex]].\n"
    )
    (notes / "engineering-hall.md").write_text(
        "The machine by the stairs reverses every page it prints.\n"
    )
    (notes / "duplex.md").write_text(
        "Flipping the switch back costs a second pass, and the stairs are\n"
        "only one flight.\n"
    )
    return notes


@pytest.fixture
def index(branch):
    idx = Index(branch=branch, store_path=branch.parent / "graph.sqlite3")
    idx.refresh()
    return idx


def test_a_question_written_in_a_note_is_proposed(index):
    """An agent walking the graph proposes the questions, so they come
    from the corpus rather than from memory."""
    found = proposals(index)

    assert [p.question for p in found] == [
        "Why does clearing a job take so long?"
    ]


def test_a_proposed_question_is_the_note_s_own_words(index):
    """Not a paraphrase of it, and not the reader's memory of it."""
    proposed = proposals(index)[0]

    assert proposed.question in index.note_text(proposed.asked_in)


def test_a_proposal_names_the_note_it_was_asked_in(index):
    assert proposals(index)[0].asked_in == "queue.md"


def test_the_notes_that_answer_are_the_ones_the_walk_reaches(index):
    """The walk is the proposer: what makes a note worth asking about is
    that the graph connects it to notes to read."""
    proposed = proposals(index)[0]

    assert proposed.answers == ["duplex.md", "engineering-hall.md"]


def test_the_note_a_question_was_asked_in_is_not_its_own_answer(index):
    """It asks the question. Counting it as support would let any
    question be kept by the note that contains it."""
    proposed = proposals(index)[0]

    assert proposed.asked_in not in proposed.answers


def test_a_question_in_a_note_the_graph_connects_to_nothing_is_not_proposed(
    tmp_path,
):
    """This finds failures of one kind, since a question is asked only if
    the graph noticed something. A question with nowhere to look is not a
    question this set is for, and proposing it would pad the set with
    questions the baseline wins by default."""
    branch = tmp_path / "notes"
    branch.mkdir()
    (branch / "alone.md").write_text("Why does nothing here have an answer?")
    (branch / "other.md").write_text("A note about a completely separate thing.")
    idx = Index(branch=branch, store_path=tmp_path / "graph.sqlite3")
    idx.refresh()

    assert proposals(idx) == []


def test_a_note_with_no_question_proposes_nothing_however_well_connected(
    tmp_path,
):
    branch = tmp_path / "notes"
    branch.mkdir()
    (branch / "a.md").write_text("The spooler restarts between jobs. See [[b]].")
    (branch / "b.md").write_text("Turning it off costs an extra pass. See [[a]].")
    idx = Index(branch=branch, store_path=tmp_path / "graph.sqlite3")
    idx.refresh()

    assert proposals(idx) == []


def test_questions_in_notes_that_hold_nothing_are_not_proposed(index):
    """The notes that answer, and the notes that do not, are separated
    before a walk ever starts."""
    everything = {p.question for p in proposals(index)}

    assert "Why does clearing a job take so long?" in everything
    assert not any("spooler" in q for q in everything)


def test_proposals_are_in_a_fixed_order_whatever_the_store_holds(index):
    """A set that came out in storage order would be a different set on
    the next run, and an evaluation over a moving set measures the
    moving."""
    assert proposals(index) == proposals(index)


# --- How much one note may say ------------------------------------------

@pytest.fixture
def crowded(tmp_path):
    """A branch where one note asks many questions and the rest ask one.

    Measured on a real branch of 131 notes: the questions came from 32
    notes, and the largest contributor held 35 of 105 -- a third of the
    set, from one note of continuous writing. Nothing was wrong with any
    of those 35, which is the problem: a set that is a third one note
    measures that note.

    Every note links to another, which is all `proposals` needs beyond a
    question, so what is being varied here is the questions and not the
    graph.
    """
    notes = tmp_path / "crowded"
    notes.mkdir()
    (notes / "diary.md").write_text(
        "See [[hall]] and [[stairs]].\n"
        "Why did the printer reverse?\n"
        "Why did the stairs jam?\n"
        "Why did the queue clear?\n"
        "Why did the duplex fail?\n"
        "Why did the spooler restart?\n"
    )
    for name, asked in (
        ("hall", "Why is the spooler talking?"),
        ("stairs", "What does the duplex need?"),
        ("border", "Where does the printer jam?"),
    ):
        (notes / f"{name}.md").write_text(
            f"See [[diary]]. The machine talks about the spooler.\n{asked}\n"
        )
    idx = Index(branch=notes, store_path=tmp_path / "graph.sqlite3")
    idx.refresh()
    return idx


def test_a_note_puts_at_most_so_many_questions_into_the_set(crowded):
    """The figure a report reads would otherwise move when one note grew
    and not at all when the graph changed."""
    per_note = Counter(p.asked_in for p in proposals(crowded))

    assert per_note["diary.md"] == DEFAULT_PER_NOTE
    assert max(per_note.values()) <= DEFAULT_PER_NOTE


def test_the_bound_picks_which_of_a_note_s_questions_are_proposed(crowded):
    """The first it asks, in the order it asks them. Ranking them by a
    proxy for how question-shaped they look would spend the note's few
    slots on whichever guess was wrong."""
    proposed = [p.question for p in proposals(crowded) if p.asked_in == "diary.md"]

    assert proposed == [
        "Why did the printer reverse?",
        "Why did the stairs jam?",
        "Why did the queue clear?",
    ]


def test_the_bound_is_on_the_set_and_not_on_the_note(crowded):
    """A caller asking for every question in the branch gets every
    question. What the bound takes away is the walk's, not the note's."""
    bounded = {p.question for p in proposals(crowded)}
    unbounded = {p.question for p in proposals(crowded, per_note=None)}

    assert "Why did the spooler restart?" not in bounded
    assert "Why did the spooler restart?" in unbounded


# --- How far the walk goes ----------------------------------------------

def test_the_walk_stops_at_its_bound(index):
    """A question is a span of notes, not a component. At no bound every
    proposal is supported by most of the corpus and the set stops being
    about whether the walk reached the right notes."""
    proposed = proposals(index, max_depth=1)[0]

    assert proposed.answers == ["duplex.md", "engineering-hall.md"]


def test_a_note_beyond_the_bound_does_not_answer(index, tmp_path):
    # No "and" before the link. `queue` and `far` both name `duplex`, so
    # that name is a term they share; add the conjunction and it is a
    # second, and the two are within a word of an edge. In four notes the
    # conjunction is in three of them, which is not yet ubiquitous, so it
    # is a term — the branch being four notes, not the walk being wrong.
    far = tmp_path / "notes" / "far.md"
    far.write_text("A note at the far end of the corridor, linking [[duplex]].")
    index.refresh()

    shallow = {p.question: p.answers for p in proposals(index, max_depth=1)}
    deep = {p.question: p.answers for p in proposals(index, max_depth=3)}

    question = "Why does clearing a job take so long?"
    assert "far.md" not in shallow[question]
    assert "far.md" in deep[question]


def test_the_bound_is_a_parameter_because_the_right_one_is_a_judgement():
    assert DEFAULT_DEPTH > 0


def test_a_limit_stops_the_walk_and_leaves_the_rest_for_another_run(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    (branch / "a.md").write_text("Why now? See [[b]].")
    (branch / "b.md").write_text("An answer. See [[a]].")
    (branch / "c.md").write_text("Why then? See [[b]].")
    idx = Index(branch=branch, store_path=tmp_path / "graph.sqlite3")
    idx.refresh()

    assert len(proposals(idx)) == 2
    assert len(proposals(idx, limit=1)) == 1


# --- Where this stops and the verdict begins -----------------------------

def test_a_proposal_claims_where_to_look_and_nothing_more(index):
    """The graph proposes; the notes decide: a question counts only when
    the notes support it, or a structure the reader invented becomes its
    own proof. A proposal that decided for itself would be that proof
    standing in for an answer, which is the failure this exists to
    prevent."""
    proposed = proposals(index)[0]

    assert isinstance(proposed, Proposal)
    assert not hasattr(proposed, "verdict")
    assert not hasattr(proposed, "kept")


def test_a_proposed_question_is_kept_when_search_cannot_follow_it(index, branch):
    """The end of the road, and the whole reason for lifting questions
    from notes rather than writing them.

    The question says "clearing a job". Neither note that answers it uses
    those words — one says "spooler", the other says "extra pass" — so
    search finds the note that asked the question and neither of the ones
    that answer it. The walk is the only way to them.
    """
    search = KeywordSearch(branch)
    proposed = proposals(index)[0]

    verdict = question_verdict(
        proposed.question, proposed.answers, search, index
    )

    assert verdict.verdict is KEPT
    assert set(verdict.supported_by) == {"duplex.md", "engineering-hall.md"}
    assert "duplex.md" not in verdict.found_by_search


def test_a_question_the_baseline_can_follow_is_left_to_the_verdict(tmp_path):
    """A proposal is not a promise. Here both answers use the question's
    own words, so search reaches them both and the verdict drops the
    question — which is what makes the set better than one built only
    from questions that were going to be kept.

    Nine padding notes, and the reason is the same as everywhere else in
    this suite: the question is *about* clearing a job, and both answers
    restate it, so its three content words are in every note of a
    three-note branch. The count calls them ubiquitous and the baseline
    refuses to search on them — which is the right answer to "which notes
    use these words", asked of three notes, and the wrong basis for a
    verdict about whether the baseline can follow a question. The padding
    states no link and shares no word with the three notes, so the walk
    still reaches the same two answers and the only thing that changes is
    how many notes the words are in.
    """
    branch = tmp_path / "notes"
    branch.mkdir()
    (branch / "queue.md").write_text(
        "## Why does clearing a job take so long?\n"
        "See [[engineering-hall]] and [[duplex]].\n"
    )
    (branch / "engineering-hall.md").write_text(
        "Clearing a job from that machine is a long wait.\n"
    )
    (branch / "duplex.md").write_text(
        "Clearing the job takes a second pass, which is a long time.\n"
    )
    for name, text in [
        ("printer.md", "printer lab duplex jams cartridge"),
        ("printer2.md", "printer lab duplex toner cartridge"),
        ("allotment.md", "allotment butt water moved committee"),
        ("allotment2.md", "allotment fence water moved shed"),
        ("ferry.md", "ferry timetable harbour crossing cancelled"),
        ("ferry2.md", "ferry harbour crossing tide delayed"),
        ("beehive.md", "beehive frames harvest lavender honey"),
        ("beehive2.md", "beehive frames harvest lavender swarm"),
        ("kayak.md", "kayak paddle river rapids practice"),
    ]:
        (branch / name).write_text(text)
    index = Index(branch=branch, store_path=tmp_path / "graph.sqlite3")
    index.refresh()
    proposed = proposals(index)[0]

    verdict = question_verdict(
        proposed.question, proposed.answers, KeywordSearch(branch), index
    )

    assert verdict.verdict is not KEPT
    assert set(proposed.answers) <= set(verdict.found_by_search)


def test_a_proposal_reads_the_store_as_it_stands(index, branch):
    """``Index.graph`` refreshes nothing, and neither does this. A walk
    that refreshed as it went would propose notes from a graph that had
    moved underneath it, and the verdict would report the difference as a
    reader inventing structure rather than as a store that was behind."""
    (branch / "new.md").write_text("Why is this not proposed yet? See [[queue]].")

    assert not any(
        p.question == "Why is this not proposed yet?" for p in proposals(index)
    )

    index.refresh()
    assert any(
        p.question == "Why is this not proposed yet?" for p in proposals(index)
    )
