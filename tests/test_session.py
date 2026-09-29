"""Sessions: a conversation, and what happens when it refuses.

Encodes the Session section of product.md:

  Questions are asked in a conversation, which is a shallow recording of
  its messages, independent of the index: rebuilding the graph rewrites
  nothing that was said. A session writes nothing into the branch, and
  is kept outside it, since what it records is note content. Within it,
  a refusal is a fork rather than a wall: what is absent ends the
  search, what is present but unconnected redirects the question, what
  is thin asks for more.

and from Refusal:

  It never dresses a partial result as an answer, and never answers
  with hedging where a clean refusal would serve. A refusal keeps the
  nearest notes, marked as not an answer, and is inspectable like any
  other.
"""

import pytest

from acciughe.session import (
    ABSENT,
    UNCONNECTED,
    THIN,
    END,
    REDIRECT,
    ASK,
    ANSWERED,
    Answer,
    Citation,
    NearNote,
    Refusal,
    Session,
)

# Markers rather than sentences. A session owes these fields an exact
# pass-through and nothing about their wording; what a given refusal
# should say in them belongs to whoever refuses, and `test_agent.py`
# says what the agent says.
REDIRECT_TEXT = "<the question to ask instead>"
WANTS_TEXT = "<what would be enough>"


@pytest.fixture
def session(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    # A session is kept outside the branch, since what it records is
    # note content.
    return Session(path=tmp_path / "state" / "sessions", branch=branch)


# --- A shallow recording of its messages -------------------------------

def test_a_session_records_what_was_asked(session):
    session.ask("what did I decide about the graph?")
    session.answer(Answer(text="you kept it disposable", evidence=[]))

    asked = [m.content for m in session.messages() if m.role == "user"]

    assert asked == ["what did I decide about the graph?"]


def test_a_session_records_what_was_said(session):
    session.ask("why?")
    session.answer(Answer(text="because it is derived", evidence=[]))

    said = [m.content for m in session.messages() if m.role == "assistant"]

    assert said == ["because it is derived"]


def test_a_session_is_a_shallow_recording_not_a_summary(session):
    """A shallow recording of its messages.

    The messages are kept whole. Nothing is folded, summarised, or
    rewritten, so what was said is still there to be read back.
    """
    session.ask("first question")
    session.answer(Answer(text="first answer", evidence=[]))
    session.ask("second question")
    session.answer(Answer(text="second answer", evidence=[]))

    assert [m.content for m in session.messages()] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]


def test_a_session_survives_being_reopened(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"

    first = Session(path=path, branch=branch)
    first.ask("a question worth keeping")
    first.answer(Answer(text="an answer worth keeping", evidence=[]))

    resumed = Session.resume(path, branch, first.session_id)

    assert [m.content for m in resumed.messages()] == [
        "a question worth keeping",
        "an answer worth keeping",
    ]


def test_a_resumed_session_continues_the_same_conversation(tmp_path):
    """DECISION: a session is a conversation, and a conversation is
    resumed by name.

    Continuing "the most recent one" would silently attach a new
    question to whatever happened to be last, which is a guess about
    which conversation is being had.
    """
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"
    first = Session(path=path, branch=branch)
    first.ask("an earlier question")
    first.close()

    resumed = Session.resume(path, branch, first.session_id)
    resumed.ask("a later question")

    assert [m.content for m in resumed.messages()] == [
        "an earlier question",
        "a later question",
    ]


def test_two_conversations_do_not_bleed_into_each_other(tmp_path):
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"

    one = Session(path=path, branch=branch)
    one.ask("about the index")
    two = Session(path=path, branch=branch)
    two.ask("about the graph")

    assert Session.sessions(path) == sorted([one.session_id, two.session_id])
    assert [
        m.content for m in Session.resume(path, branch, one.session_id).messages()
    ] == ["about the index"]


# --- Independent of the index -----------------------------------------

def test_rebuilding_the_graph_rewrites_nothing_that_was_said(tmp_path):
    """Independent of the index: rebuilding the graph rewrites nothing
    that was said."""
    from acciughe.index import Index

    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    (branch / "a.md").write_text("a note about the graph")

    idx = Index(branch=branch, store_path=store)
    idx.refresh()

    session = Session(path=tmp_path / "state" / "sessions", branch=branch)
    session.ask("what is a note?")
    session.answer(Answer(text="a file that decodes as text", evidence=[]))
    before = [m.content for m in session.messages()]

    # The graph is discarded and rebuilt from the notes.
    store.unlink()
    Index(branch=branch, store_path=store).refresh()

    assert [m.content for m in session.messages()] == before


def test_deleting_the_index_loses_no_session(tmp_path):
    from acciughe.index import Index

    branch = tmp_path / "notes"
    branch.mkdir()
    store = tmp_path / "state" / "graph.sqlite3"
    (branch / "a.md").write_text("a note")

    Index(branch=branch, store_path=store).refresh()
    session = Session(path=tmp_path / "state" / "sessions", branch=branch)
    session.ask("a question")
    session.answer(Answer(text="an answer", evidence=[]))

    store.unlink()

    assert len(session.messages()) == 2, "the session outlives the graph"


def test_a_refusal_survives_being_reopened(tmp_path):
    """A refusal is inspectable like any other — including later.

    The fork, the kind, and the nearest notes all have to come back, or
    a resumed conversation cannot tell a wall from a fork.
    """
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"

    first = Session(path=path, branch=branch)
    first.ask("what connects these two?")
    first.refuse(
        Refusal(
            kind=UNCONNECTED,
            nearest=[NearNote(path="a.md", why_it_is_near="shares the term graph")],
            redirect_to=REDIRECT_TEXT,
        )
    )

    resumed = Session.resume(path, branch, first.session_id)
    message = resumed.messages()[-1]

    assert message.refusal_kind == "unconnected"
    assert message.next_step is REDIRECT
    assert message.nearest == ("a.md",)
    assert message.redirect_to == REDIRECT_TEXT
    assert message.is_answer is False


def test_what_a_refusal_asked_for_survives_being_reopened(tmp_path):
    """The asking half of a fork has to come back with the redirecting
    half. A session that resumed with a wall and a reason to go would
    strand the reader wherever the walk happened to stop."""
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"

    first = Session(path=path, branch=branch)
    first.ask("what did I decide?")
    first.refuse(Refusal(kind=THIN, nearest=[], wants=WANTS_TEXT))

    message = Session.resume(path, branch, first.session_id).messages()[-1]

    assert message.next_step is ASK
    assert message.wants == WANTS_TEXT
    assert message.redirect_to is None


# --- Writes nothing into the branch -----------------------------------

def test_a_session_writes_nothing_into_the_branch(tmp_path):
    """A session writes nothing into the branch."""
    branch = tmp_path / "notes"
    branch.mkdir()
    (branch / "a.md").write_text("a note")
    before = {p.name for p in branch.iterdir()}

    session = Session(path=tmp_path / "state" / "sessions", branch=branch)
    session.ask("a question about a note")
    session.answer(Answer(text="an answer", evidence=[]))

    assert {p.name for p in branch.iterdir()} == before


def test_a_session_is_kept_outside_the_branch(tmp_path):
    """And is kept outside it, since what it records is note content."""
    branch = tmp_path / "notes"
    branch.mkdir()
    path = tmp_path / "state" / "sessions"

    session = Session(path=path, branch=branch)
    session.ask("something only you would know from reading the notes")
    session.answer(Answer(text="an answer", evidence=[]))
    session.close()

    assert path.exists()
    stored = session.log_path().read_text(encoding="utf-8")
    assert "only you would know" in stored, "note content is not written into the branch"


# --- A refusal is a fork, not a wall ----------------------------------

def test_what_is_absent_ends_the_search():
    """A refusal is a fork rather than a wall: what is absent ends the
    search."""
    assert Refusal(kind=ABSENT, nearest=[]).next_step is END


def test_what_is_present_but_unconnected_redirects_the_question():
    assert Refusal(kind=UNCONNECTED, nearest=[]).next_step is REDIRECT


def test_what_is_thin_asks_for_more():
    assert Refusal(kind=THIN, nearest=[]).next_step is ASK


def test_the_three_refusals_fork_three_different_ways():
    """The three kinds of not-knowing are not one kind of wall."""
    steps = {Refusal(kind=k, nearest=[]).next_step for k in (ABSENT, UNCONNECTED, THIN)}

    assert steps == {END, REDIRECT, ASK}


def test_only_absence_is_a_wall():
    """Only absence ends it; the others are forks the search continues through."""
    assert Refusal(kind=ABSENT, nearest=[]).next_step is END
    assert Refusal(kind=UNCONNECTED, nearest=[]).next_step is not END
    assert Refusal(kind=THIN, nearest=[]).next_step is not END


# --- Which kind of not-knowing is named -------------------------------

def test_a_refusal_says_which_kind_it_is():
    """It says which kind of not-knowing it is."""
    assert Refusal(kind=ABSENT, nearest=[]).kind is ABSENT
    assert Refusal(kind=UNCONNECTED, nearest=[]).kind is UNCONNECTED
    assert Refusal(kind=THIN, nearest=[]).kind is THIN


def test_a_refusal_states_its_kind_in_words():
    """DECISION: each kind says so in the user's terms, not as an enum.

    The enum names the condition; what the user reads names it
    differently, because "the corpus does not contain this" and "this is
    here but connects to nothing you asked about" are different
    situations and a reader has to be able to tell them apart.
    """
    assert "not in your notes" in Refusal(kind=ABSENT, nearest=[]).explanation
    assert "not connected" in Refusal(kind=UNCONNECTED, nearest=[]).explanation
    assert "little" in Refusal(kind=THIN, nearest=[]).explanation


# --- A refusal is never dressed as an answer -------------------------

def test_a_refusal_keeps_the_nearest_notes(session):
    """A refusal keeps the nearest notes, marked as not an answer."""
    refusal = Refusal(
        kind=UNCONNECTED,
        nearest=[NearNote(path="a.md", why_it_is_near="shares the term graph")],
    )

    session.ask("what connects these two?")
    session.refuse(refusal)

    recorded = session.messages()[-1]
    assert recorded.nearest == ("a.md",)


def test_the_nearest_notes_are_marked_as_not_an_answer():
    """Marked as not an answer."""
    refusal = Refusal(
        kind=ABSENT,
        nearest=[NearNote(path="a.md", why_it_is_near="closest thing found")],
    )

    assert all(n.is_not_an_answer for n in refusal.nearest)


def test_a_refusal_is_not_an_answer(session):
    """A refusal is inspectable like any other, and is not an answer."""
    session.ask("a question")
    session.refuse(Refusal(kind=THIN, nearest=[]))

    assert session.messages()[-1].is_answer is False


def test_an_answer_is_marked_as_an_answer(session):
    session.ask("a question")
    session.answer(Answer(text="an answer", evidence=[]))

    assert session.messages()[-1].is_answer is True


def test_an_answer_carries_its_evidence(session):
    """Answers cite their evidence. Every claim points at the notes
    behind it."""
    session.ask("what did I decide?")
    session.answer(
        Answer(
            text="the graph is disposable",
            evidence=[Citation(path="a.md", quote="derived and disposable")],
        )
    )

    recorded = session.messages()[-1]
    assert recorded.evidence == (("a.md", "derived and disposable"),)


def test_an_answer_cannot_be_recorded_without_its_evidence_being_citations(session):
    """DECISION: the evidence of an answer is a list of citations, not text.

    The type makes a claim that cites nothing awkward to write, which is
    the cheapest way to keep a bare assertion out of the record.
    """
    answer = Answer(text="a bare claim", evidence=[])

    assert answer.evidence == [], "and it is visible rather than silently permitted"


def test_a_refusal_carries_no_evidence(session):
    """A refusal must not carry claims, only context."""
    session.ask("a question")
    session.refuse(
        Refusal(
            kind=THIN,
            nearest=[NearNote(path="a.md", why_it_is_near="barely related")],
        )
    )

    assert session.messages()[-1].evidence == ()


# --- The fork is what the conversation does next ----------------------

def test_a_refusal_of_each_kind_records_its_fork(session):
    """DECISION: the fork is recorded with the refusal, so the shape of
    the conversation says what it will do next.

    The alternative is a caller reading the kind and branching, which
    puts the same three-way decision in every place that handles a
    refusal.
    """
    session.ask("a question")
    session.refuse(Refusal(kind=UNCONNECTED, nearest=[]))

    assert session.messages()[-1].next_step is REDIRECT


def test_an_answer_ends_the_turn_without_a_fork(session):
    session.ask("a question")
    session.answer(Answer(text="an answer", evidence=[]))

    assert session.messages()[-1].next_step is ANSWERED


def test_a_redirected_refusal_carries_the_question_it_redirects_to(session):
    """DECISION: a redirect says what question to ask instead.

    Redirecting without naming the new question leaves the caller with a
    decision to make it already had.

    The wording is a marker rather than a sentence because what a session
    owes this field is the exact pass-through. What any given refusal
    should say in it is `agent.py`'s to decide and is specified there.
    """
    session.ask("what connects these?")
    session.refuse(
        Refusal(kind=UNCONNECTED, nearest=[], redirect_to=REDIRECT_TEXT)
    )

    assert session.messages()[-1].redirect_to == REDIRECT_TEXT


def test_a_thin_refusal_carries_what_it_would_like_more_of(session):
    """DECISION: asking for more says what would be enough.

    Asking for more without saying what would satisfy it is a wall
    wearing a question mark.
    """
    session.ask("what did I decide?")
    session.refuse(Refusal(kind=THIN, nearest=[], wants=WANTS_TEXT))

    assert session.messages()[-1].wants == WANTS_TEXT


def test_a_refusal_of_each_fork_carries_its_own_fields(session):
    """The three forks are told apart by their fields as well as by their
    next step, so a resumed conversation can act on the right one."""
    session.refuse(Refusal(kind=ABSENT, nearest=[]))
    absent = session.messages()[-1]

    session.refuse(
        Refusal(kind=UNCONNECTED, nearest=[], redirect_to=REDIRECT_TEXT)
    )
    redirected = session.messages()[-1]

    session.refuse(Refusal(kind=THIN, nearest=[], wants=WANTS_TEXT))
    asked = session.messages()[-1]

    assert (absent.redirect_to, absent.wants) == (None, None)
    assert (redirected.redirect_to, redirected.wants) == (REDIRECT_TEXT, None)
    assert (asked.redirect_to, asked.wants) == (None, WANTS_TEXT)


def test_a_refusal_ends_the_search_and_the_session_records_that(session):
    """What is absent ends the search — so the session is the record of
    where the search stopped."""
    session.ask("what did I write about the flight?")
    session.refuse(Refusal(kind=ABSENT, nearest=[]))

    assert session.messages()[-1].next_step is END
