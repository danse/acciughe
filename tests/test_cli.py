"""The command.

The gap this file closes. Every other test in the project reaches a
function; none of them could have caught a command whose subcommand was
never wired, whose arguments were not parsed, or which wrote a store
into somebody's notes. `main` is the whole of that path, and it was
unwritten until now.

product.md:

  Questions are asked in a conversation, which is a shallow recording of
  its messages, independent of the index: rebuilding the graph rewrites
  nothing that was said. A session writes nothing into the branch, and
  is kept outside it, since what it records is note content.

  A branch is indexed before it is answered from, and re-read when it no
  longer matches. [...] Both happen unasked, and are reported rather than
  silent.

  Notes are the source of truth. They are never rewritten, moved or
  interpreted in place.

  A graph is derived and disposable. It is rebuilt from the notes and can
  be deleted at any time without loss.

The model is a real HTTP server on a real port that replies with a
recorded reply, as in `test_model.py`. A stubbed `answer_from` would let
the command be wrong about the model in ways no test here could see.
"""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from acciughe.cli import main, sessions_for, store_for
from acciughe.session import Session

# What gemma3:270m said when asked about this corpus, recorded verbatim.
# The answer is the whole sentence the note holds, and the two notes are
# numbered over the evidence the walk offers — nearest first, so
# compiler.md, garden.md, index.md. It numbered 1 and 3 because garden.md
# has nothing in it about the pipeline.
ANSWERED = json.dumps(
    {
        "answer": "The compiler pipeline reads the notes folder, builds an "
                  "index, and writes the store.",
        "note1": 1,
        "quote1": "The compiler pipeline reads the notes folder, builds an "
                  "index, and writes the store.",
        "note2": 3,
        "quote2": "The index is derived and disposable.",
    }
)

INVENTED = json.dumps(
    {
        "answer": "The pipeline reads, indexes and caches, which is the usual "
                  "arrangement for this kind of tool.",
        "note1": 1,
        "quote1": "The compiler pipeline reads the notes folder, builds an "
                  "index, and writes the store.",
        "note2": 3,
        "quote2": "The index is derived and disposable.",
    }
)


class _Model(BaseHTTPRequestHandler):
    """A real socket that is not a model."""

    reply: str = ANSWERED

    def do_POST(self):  # noqa: N802 - the name is BaseHTTPRequestHandler's
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        payload = json.dumps({"message": {"content": type(self).reply}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def model():
    _Model.reply = ANSWERED
    running = HTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{running.server_port}"
    running.shutdown()


@pytest.fixture
def branch(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "compiler.md").write_text(
        "The compiler pipeline reads the notes folder, builds an index, and "
        "writes the store. See [[index]] and [[garden]].\n"
    )
    (notes / "index.md").write_text(
        "The index is derived and disposable. Deleting the store costs "
        "nothing because every row comes from the notes.\n"
    )
    (notes / "garden.md").write_text(
        "The allotment committee moved the water butt to the far end. "
        "Nothing has been dry there since. See [[index]].\n"
    )
    return notes


@pytest.fixture
def state(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def questioned(tmp_path):
    """A branch with a question in it that search cannot follow.

    The walk reaches `tray.md` and `roller.md` by links alone; neither
    shares a word with the question, so the verdict keeps it and the run
    has something to ask. The corpus above has no questions in it at all,
    which is the right shape for testing `ask` and the wrong one for
    testing a run.
    """
    notes = tmp_path / "questioned"
    notes.mkdir()
    (notes / "press.md").write_text(
        "Fed from a tray. See [[tray]].\nWhy did the press jam?"
    )
    (notes / "tray.md").write_text("Damp stock, weighed. See [[roller]].")
    (notes / "roller.md").write_text("Grip is lost by August. See [[paper]].")
    (notes / "paper.md").write_text("Order by bale, not ream.")
    return notes


# gemma3:270m answering the press question: one quotation of the note the
# walk starts from, which is true, and one invented line attributed to
# the note after it, which is the mis-attribution the guard exists for.
PRESSED = json.dumps(
    {
        "answer": "Fed from a tray.",
        "note1": 1,
        "quote1": "Fed from a tray.",
        "note2": 2,
        "quote2": "The tray was flushed every Friday.",
    }
)


def run(argv, out=None, err=None):
    """The command, with its streams replaceable so it can be tested
    without a terminal."""
    out = out if out is not None else io.StringIO()
    err = err if err is not None else io.StringIO()
    return main(argv, out, err), out, err


def ask(branch, state, model, question, *extra):
    return run(
        ["--branch", str(branch), "--state", str(state),
         "--host", model, "ask", question, *extra]
    )


# --- Asking -------------------------------------------------------------

def test_a_question_asked_through_the_command_is_answered(branch, state, model):
    code, out, _err = ask(branch, state, model, "what does the pipeline read?")

    assert code == 0
    assert "reads the notes folder, builds an index, and writes the store." \
        in out.getvalue()
    assert 'compiler.md: "' in out.getvalue()
    assert 'index.md: "' in out.getvalue()
    assert "These are notes, not an answer." not in out.getvalue()


def test_a_sentence_the_notes_do_not_contain_is_not_shown(branch, state, model):
    _Model.reply = INVENTED

    code, out, _err = ask(branch, state, model, "what does the pipeline read?")

    assert code == 0
    assert "which is the usual arrangement" not in out.getvalue()
    assert "These are notes, not an answer." in out.getvalue()


def test_the_progress_goes_to_stderr_and_the_answer_to_stdout(
    branch, state, model
):
    """The turn pauses for a minute or two, and a reader needs to know the
    tool is working. But `acciughe ask ... | less` should show the answer
    and not the narration of how it got there — so the two streams are
    kept apart rather than interleaved on one."""
    _code, out, err = ask(branch, state, model, "what does the pipeline read?")

    assert "the model is writing" in err.getvalue()
    assert "the model is writing" not in out.getvalue()
    assert "reads the notes folder, builds an index, and writes the store." \
        in out.getvalue()


def test_a_question_nothing_in_the_corpus_uses_is_refused(branch, state, model):
    code, out, err = ask(branch, state, model, "what is the escape velocity "
                         "of a duck?")

    assert code == 0
    assert "This is not in your notes." in out.getvalue()
    assert "the model is writing" not in err.getvalue()


# --- The branch is never written to -------------------------------------

def test_asking_writes_nothing_into_the_branch(branch, state, model):
    """product.md: "Notes are the source of truth. They are never
    rewritten, moved or interpreted in place."

    Both the graph and the conversation are kept out of it, and the
    reason is not tidiness: what a session records is note content, and a
    branch is a directory somebody's notes live in.
    """
    before = sorted((p.name, p.read_text()) for p in branch.iterdir())

    ask(branch, state, model, "what does the pipeline read?")

    assert sorted((p.name, p.read_text()) for p in branch.iterdir()) == before
    assert not list(branch.glob("**/*.sqlite3"))
    assert not list(branch.glob("**/*.jsonl"))


def test_the_graph_and_the_conversations_live_outside_the_branch(
    branch, state, model
):
    ask(branch, state, model, "what does the pipeline read?")

    store = store_for(branch.resolve(), state)
    sessions = sessions_for(branch.resolve(), state)
    assert store.exists()
    assert store.is_relative_to(state)
    assert sessions.is_relative_to(state)
    assert not store.is_relative_to(branch)
    assert not sessions.is_relative_to(branch)


def test_two_branches_never_share_a_graph(tmp_path, state, model):
    """The failure this guards is silent and would be about the reader's
    own notes: one store holding two branches' graphs answers a question
    from notes that are not there, and every citation would check out
    because the notes are real — just not these ones."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    for notes in (first, second):
        notes.mkdir()
        (notes / "compiler.md").write_text(
            "The compiler pipeline reads the notes folder, builds an index, "
            "and writes the store. See [[index]] and [[garden]].\n"
        )
        (notes / "index.md").write_text("The index is derived and disposable.\n")
        (notes / "garden.md").write_text(
            "The allotment committee moved the water butt to the far end.\n"
        )

    assert store_for(first, state) != store_for(second, state)


def test_two_branches_named_the_same_thing_never_share_a_graph(tmp_path, state):
    """A digest of the path, not of the name: two branches can both be
    called ``notes``."""
    first = tmp_path / "one" / "notes"
    second = tmp_path / "two" / "notes"

    assert store_for(first, state) != store_for(second, state)


# --- Conversations ------------------------------------------------------

def test_questions_are_asked_in_one_conversation(branch, state, model):
    """product.md: "Questions are asked in a conversation." Asking twice
    without naming one continues the same, rather than starting a new
    one per question."""
    ask(branch, state, model, "what does the pipeline read?")
    ask(branch, state, model, "what did the allotment committee do?")

    held = Session.sessions(sessions_for(branch.resolve(), state))
    assert len(held) == 1
    assert len(
        Session(path=sessions_for(branch.resolve(), state),
                branch=branch, session_id=held[0]).reopened().messages()
    ) == 4


def test_naming_a_conversation_starts_a_different_one(branch, state, model):
    sessions = sessions_for(branch.resolve(), state)

    ask(branch, state, model, "what does the pipeline read?", "--session", "a")
    ask(branch, state, model, "what does the pipeline read?", "--session", "b")

    assert Session.sessions(sessions) == ["a", "b"]


def test_the_sessions_command_lists_what_there_is(branch, state, model):
    ask(branch, state, model, "what does the pipeline read?", "--session", "a")

    code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "sessions"]
    )

    assert code == 0
    assert out.getvalue().startswith("a  ")
    assert "message" in out.getvalue()


def test_the_sessions_command_says_so_when_there_are_none(branch, state):
    """An empty list reads as a broken command. Saying it is empty is the
    difference between "you have no conversations" and "this does not
    work"."""
    code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "sessions"]
    )

    assert code == 0
    assert "no conversations" in out.getvalue()


def test_rebuilding_the_graph_rewrites_nothing_that_was_said(
    branch, state, model
):
    """product.md: "independent of the index: rebuilding the graph
    rewrites nothing that was said."""
    ask(branch, state, model, "what does the pipeline read?", "--session", "a")
    sessions = sessions_for(branch.resolve(), state)
    before = (sessions / "a.jsonl").read_text()

    run(["--branch", str(branch), "--state", str(state), "index"])
    (branch / "garden.md").unlink()
    run(["--branch", str(branch), "--state", str(state), "index"])

    assert (sessions / "a.jsonl").read_text() == before


# --- The read on its own -----------------------------------------------

def test_the_index_command_reports_the_read(branch, state):
    code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "index"]
    )

    assert code == 0
    assert "read 3 notes" in out.getvalue()


def test_a_read_that_changed_nothing_says_so(branch, state):
    run(["--branch", str(branch), "--state", str(state), "index"])

    code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "index"]
    )

    assert "up to date" in out.getvalue()


def test_the_graph_command_reports_the_three_measurements_the_spec_names(branch, state):
    """product.md: "The graph is measured rather than assumed — its
    density, how much of it is co-occurrence rather than relation, how
    many notes end up connected to nothing."

    All three were computed by `graph_metrics` and none of them could be
    seen, which is a spec satisfied in a test and not in a hand. This is
    the command that closes that, and the gap this file exists to catch:
    a measurement nothing renders is a measurement nobody has.
    """
    code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "graph"]
    )

    said = out.getvalue()

    assert code == 0
    assert "3 notes," in said
    assert "relations to a note on average" in said, "its density"
    assert "co-occurrence" in said, "how much is co-occurrence"
    assert "stated links" in said, "and how much is stated relation"
    assert "connect to nothing" in said, "and how many notes reach nothing"


def test_the_graph_command_reads_the_branch_before_it_measures_it(branch, state):
    """The read is above the figures and not suppressed.

    These are measurements *of a graph*. A graph that had drifted from the
    branch would be measured and reported as though it were this one, and
    the reader would have no way to tell — the only defence is saying what
    was read on the same screen, which is also what "reported rather than
    silent" asks for.
    """
    run(["--branch", str(branch), "--state", str(state), "index"])
    (branch / "newcomer.md").write_text("A note that arrived after the read.")

    _code, out, _err = run(
        ["--branch", str(branch), "--state", str(state), "graph"]
    )

    said = out.getvalue()

    assert said.startswith("notes: "), "the read is reported, not assumed"
    assert "1 new" in said, "so a reader knows the graph was just rebuilt"
    assert "4 notes," in said, "and the figures are of the branch as it now stands"


def test_measuring_the_graph_of_a_path_that_is_not_a_branch_says_so(tmp_path, state):
    code, out, err = run(
        ["--branch", str(tmp_path / "nowhere"), "--state", str(state), "graph"]
    )

    assert code == 1
    assert "not a branch" in err.getvalue()
    assert out.getvalue() == ""


# --- Measuring the branch -----------------------------------------------

def test_the_evaluate_command_prints_what_a_run_found(questioned, state, model):
    _Model.reply = PRESSED
    code, out, err = run(
        ["--branch", str(questioned), "--state", str(state),
         "--host", model, "evaluate"]
    )

    said = out.getvalue()
    assert code == 0
    assert "1 of 1 proposed question kept" in said
    assert "answers 0 of 1 asked" in said
    assert "1 shown as parts" in said, (
        "the guard dropped the invented quotation and kept the true one"
    )
    assert "citations 1 of 2 in the note they name (50%)" in said
    assert "tray.md" in said, "and the invented line is named"


def test_a_run_that_never_got_to_a_question_says_it_was_never_reached(
    questioned, state, model
):
    _Model.reply = PRESSED
    code, out, _err = run(
        ["--branch", str(questioned), "--state", str(state),
         "--host", model, "evaluate", "--limit", "0"]
    )

    assert code == 0
    assert "1 of 1 proposed question kept, 1 judged and not reached" in out.getvalue()
    assert "answers 0 of 0 asked" in out.getvalue(), (
        "and nothing was asked, so nothing was answered or cited"
    )


def test_a_run_asks_no_questions_and_records_no_conversation(questioned, state, model):
    """A trial is a measurement of the notes, not a conversation with them.

    A session is a record of what a person asked. Filing a question the
    tool asked itself in the same log would put it among the questions
    the reader asked, which is the one thing the two have to be told
    apart by.
    """
    _Model.reply = PRESSED
    run(["--branch", str(questioned), "--state", str(state),
         "--host", model, "evaluate"])

    assert Session.sessions(sessions_for(questioned, state)) == []
    assert not (questioned / "sessions").exists(), (
        "and nothing was written into the branch"
    )


def test_measuring_a_path_that_is_not_a_branch_says_so(tmp_path, state, model):
    code, out, err = run(
        ["--branch", str(tmp_path / "nowhere"), "--state", str(state),
         "--host", model, "evaluate"]
    )

    assert code == 1
    assert "not a branch" in err.getvalue()
    assert out.getvalue() == ""


def test_a_run_says_what_it_has_found_so_far_while_it_goes_on(questioned, state, model):
    """A turn is a minute or two, so the report at the end is not the only
    thing a person sees.

    A real branch proposes enough questions to make a run an evening, and
    a reader watching one has no way to know whether it is finding
    anything — least of all whether the model is still inventing its
    quotes, which is the finding the whole thing exists for. The progress
    goes to stderr because it is progress; the report is the answer, and
    the two are not in the same stream.
    """
    _Model.reply = PRESSED
    code, out, err = run(
        ["--branch", str(questioned), "--state", str(state),
         "--host", model, "evaluate"]
    )

    progress = err.getvalue()

    assert code == 0
    assert "[1] press.md: " in progress, "each receipt as it was decided"
    assert "parts, not an answer" in progress
    assert "1 of 2 quotes invented" in progress, (
        "and the fabrication is on the line while there is still a run to stop"
    )
    assert "quoted" not in progress, "the line carries the count, not the prose"
    assert out.getvalue().startswith("1 of 1 proposed question kept"), (
        "and the report is the only thing on stdout"
    )


# --- Things that are not a branch --------------------------------------

def test_asking_a_path_that_is_not_a_branch_says_so(tmp_path, state, model):
    code, out, err = ask(tmp_path / "nowhere", state, model, "anything?")

    assert code == 1
    assert "not a branch" in err.getvalue()
    assert out.getvalue() == ""


def test_a_question_is_required(tmp_path, state, model):
    with pytest.raises(SystemExit):
        run(["--branch", str(tmp_path), "--state", str(state), "--host", model,
             "ask"])
