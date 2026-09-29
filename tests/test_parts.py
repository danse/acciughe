"""The third outcome: notes the walk found, shown instead of a sentence.

Added to `test_agent.py`'s territory and kept separate because it is a
different claim. `test_agent.py` pins that a turn answers or refuses; this
file pins the case where neither is honest, and that the tool says so by
showing what it found instead of writing a sentence over it.

product.md:

  It never dresses a partial result as an answer, and never answers with
  hedging where a clean refusal would serve. A refusal keeps the nearest
  notes, marked as not an answer, and is inspectable like any other. An
  answer is given only when the evidence supports it.

  When the graph does not support an answer, the command says so, and
  says which kind of not-knowing it is: absent from the corpus, present
  but unconnected, or thin.

The measured reason this outcome exists: `gemma3:270m` copies a sentence
out of a note when it can answer, and writes a new one when it cannot.
The new one is the only part of a reply no check reaches. The quoted
spans beside it are still verbatim and still attributable, so the walk's
work is worth keeping even when the sentence is not.

**Not a refusal, and not an answer.** It is neither of the three kinds of
not-knowing, because the graph did not fail to know — it reached notes
and read them. Calling it thin would be wrong twice: thin means there was
too little to answer from, and here there was enough for the model to
write something. That something is simply not shown.
"""

import pytest

from acciughe.agent import turn
from acciughe.session import (
    Answer,
    Citation,
    Components,
    Next,
    Refusal,
    RefusalKind,
    Session,
)

ASKING = "what does the compiler pipeline read?"


@pytest.fixture
def branch(tmp_path):
    """A seed note the question lands on, and two notes the walk reaches.

    The two are deliberately worded differently and share only three
    terms. Identical bodies would make a mis-attribution undetectable,
    which is the one thing the citation check is for.
    """
    notes = tmp_path / "parts"
    notes.mkdir()
    (notes / "seed.md").write_text(
        "compiler pipeline corpus. See [[one]] and [[two]]."
    )
    (notes / "one.md").write_text("branch notes about unrelated gardening")
    (notes / "two.md").write_text("branch notes about allotment planning")
    return notes


@pytest.fixture
def index(branch):
    from acciughe.index import Index

    idx = Index(branch=branch, store_path=branch.parent / "graph.sqlite3")
    idx.refresh()
    return idx


def parts(*pairs):
    """A phraser that sets the sentence aside and returns the parts."""
    def phraser(question, evidence):
        return Components(
            evidence=[Citation(path=p, quote=q) for p, q in pairs]
        )

    return phraser


# --- The turn -----------------------------------------------------------

def test_the_parts_are_what_the_reader_is_shown(index):
    result = turn(
        ASKING, index,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    assert isinstance(result.outcome, Components)
    assert result.outcome.evidence == [
        Citation(path="one.md", quote="about unrelated gardening")
    ]


def test_the_parts_are_not_an_answer(index):
    """product.md: "It never dresses a partial result as an answer." The
    type has to be able to say so without the reader checking."""
    result = turn(
        ASKING, index,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    assert not isinstance(result.outcome, Answer)
    assert result.outcome.is_an_answer is False


def test_the_parts_carry_no_sentence_to_dress_them_up(index):
    """There is no field to put one in. A `text` that defaulted to empty
    would be one somebody would eventually fill."""
    result = turn(
        ASKING, index,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    assert not hasattr(result.outcome, "text")


def test_a_turn_of_parts_still_keeps_what_the_phraser_proposed(index):
    """Otherwise citation correctness is unmeasurable: a guard that
    rewrote history would make every answer look perfect by
    construction."""
    proposed = parts(("one.md", "about unrelated gardening"), ("two.md", "nothing"))

    result = turn(ASKING, index, answer_from=proposed)

    assert [c.path for c in result.proposed.evidence] == ["one.md", "two.md"]


# --- Falling back from an answer ----------------------------------------

def answering(*pairs):
    """A phraser that writes a sentence over its citations."""
    def phraser(question, evidence):
        return Answer(
            text="The pipeline reads the notes folder, and the index is "
                 "derived and disposable.",
            evidence=[Citation(path=p, quote=q) for p, q in pairs],
        )

    return phraser


def test_one_bad_citation_falls_back_to_the_parts_that_checked_out(index):
    """Measured, on the first live turn: the model wrote a correct answer
    with two real quotes, attributed one of them to the wrong note, and
    the guard refused the whole turn — throwing away a citation that was
    perfectly real along with the one that was not.

    The sentence is not shown, so the reason an answer has to refuse
    itself whole is gone. Nothing is left standing on nothing, because
    nothing is being claimed: what remains is one note saying one thing.
    """
    result = turn(
        ASKING, index,
        answer_from=answering(
            ("one.md", "about unrelated gardening"),
            ("two.md", "about allotment planning, and then some weather"),
        ),
    )

    assert isinstance(result.outcome, Components)
    assert [c.path for c in result.outcome.evidence] == ["one.md"]


def test_a_fabrication_cannot_be_washed_out_by_falling_back(index):
    """The property that makes the fallback safe. Measured: asked about
    the weather with two notes about a compiler, the model invented a
    sentence, attributed it to both notes, and did not decline.

    Every citation fails, so nothing is left to show and the turn is a
    refusal. Showing a fabrication is not possible, because the only
    thing ever shown is a quote that was verified against the note it
    names.
    """
    result = turn(
        ASKING, index,
        answer_from=answering(
            ("one.md", "The weather in Lisbon was cloudy and sunny."),
            ("two.md", "The weather in Lisbon was cloudy and sunny."),
        ),
    )

    assert isinstance(result.outcome, Refusal)
    assert result.outcome.kind is RefusalKind.THIN
    assert result.citations.correct == 0


def test_falling_back_still_records_what_the_model_proposed(index):
    """Otherwise the number is a measurement of the filter rather than of
    the model, and a model that mis-attributes every other quote would
    score the same as one that never quotes at all."""
    result = turn(
        ASKING, index,
        answer_from=answering(
            ("one.md", "about unrelated gardening"),
            ("two.md", "about allotment planning, and then some weather"),
        ),
    )

    assert result.citations.total == 2
    assert result.citations.correct == 1
    assert [f.path for f in result.citations.fabricated] == ["two.md"]
    assert result.proposed.text.startswith("The pipeline reads")


# --- Dropping the parts that do not check out ---------------------------

def test_a_part_that_is_not_in_its_note_is_dropped(index):
    """The opposite of the answer rule, and the reason this is a separate
    type: a part asserts only itself, so the parts beside it are still
    true without it."""
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "about unrelated gardening"),
            ("two.md", "the weather in Lisbon was sunny"),
        ),
    )

    assert [c.path for c in result.outcome.evidence] == ["one.md"]


def test_a_note_named_that_does_not_exist_drops_only_its_own_part(index):
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "about unrelated gardening"),
            ("nowhere.md", "about unrelated gardening"),
        ),
    )

    assert [c.path for c in result.outcome.evidence] == ["one.md"]


def test_a_quote_misattributed_is_dropped_even_when_it_is_really_in_the_note(
    index,
):
    """Measured: the model took a real quote out of one note and named
    another. The quote is true and the citation is not, and the citation
    is what the reader follows."""
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "about unrelated gardening"),
            ("two.md", "about unrelated gardening"),
        ),
    )

    assert [c.path for c in result.outcome.evidence] == ["one.md"]


def test_one_good_part_survives_the_dropping_of_a_bad_neighbour(index):
    """A note can be cited twice, once right and once not. Dropping the
    note would lose the true quote as well, which is why a finding
    carries the quote and not only the path."""
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "about unrelated gardening"),
            ("one.md", "something the note does not say"),
        ),
    )

    assert [c.quote for c in result.outcome.evidence] == [
        "about unrelated gardening"
    ]


def test_the_report_still_counts_everything_the_model_proposed(index):
    """The number is a measurement of the model, so it is taken before
    anything is dropped. Reporting only the survivors would make a model
    that quotes badly look like one that quotes once."""
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "about unrelated gardening"),
            ("two.md", "the weather in Lisbon was sunny"),
        ),
    )

    assert result.citations.total == 2
    assert result.citations.correct == 1
    assert len(result.outcome.evidence) == 1


# --- When there is nothing left to show ---------------------------------

def test_no_parts_at_all_is_a_refusal(index):
    """An empty set of parts is a clean refusal, not a turn that shows
    the reader nothing and calls it a result."""
    result = turn(ASKING, index, answer_from=parts())

    assert isinstance(result.outcome, Refusal)
    assert result.outcome.kind is RefusalKind.THIN


def test_parts_where_none_check_out_is_a_refusal(index):
    result = turn(
        ASKING, index,
        answer_from=parts(
            ("one.md", "the weather in Lisbon was sunny"),
            ("two.md", "clouds over the estuary"),
        ),
    )

    assert isinstance(result.outcome, Refusal)
    assert result.outcome.kind is RefusalKind.THIN


def test_a_refusal_from_dropped_parts_keeps_the_notes_it_walked_to(index):
    """product.md: "A refusal keeps the nearest notes, marked as not an
    answer, and is inspectable like any other." Dropping every part must
    not throw away the fact that the walk got somewhere."""
    result = turn(
        ASKING, index,
        answer_from=parts(("one.md", "the weather in Lisbon was sunny")),
    )

    assert [n.path for n in result.outcome.nearest] == [
        "seed.md", "one.md", "two.md"
    ]


# --- In a session -------------------------------------------------------

def test_a_session_records_the_parts_without_calling_them_an_answer(
    index, tmp_path
):
    session = Session(path=tmp_path / "sessions", branch=index.branch)

    turn(
        ASKING, index, session=session,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    recorded = session.messages()[-1]
    assert recorded.is_answer is False
    assert recorded.refusal_kind is None
    assert recorded.evidence == (("one.md", "about unrelated gardening"),)


def test_a_reopened_session_tells_parts_from_a_refusal(index, tmp_path):
    """Three things can come back from a turn, and after the conversation
    is closed and reopened the log is the only record of which. A reader
    resuming a session has to be able to tell a conclusion from a
    not-knowing from a set of notes."""
    session = Session(path=tmp_path / "sessions", branch=index.branch)
    turn(
        ASKING, index, session=session,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )
    turn(ASKING, index, session=session, answer_from=parts())

    last_two = session.reopened().messages()[-2:]

    assert last_two[0].refusal_kind is None
    assert last_two[0].next_step is Next.ANSWERED
    assert last_two[1].refusal_kind == RefusalKind.THIN.value


def test_a_session_writes_the_parts_into_the_branch_never(index, tmp_path):
    before = sorted(p.name for p in index.branch.iterdir())
    session = Session(path=tmp_path / "sessions", branch=index.branch)

    turn(
        ASKING, index, session=session,
        answer_from=parts(("one.md", "about unrelated gardening")),
    )

    assert sorted(p.name for p in index.branch.iterdir()) == before
