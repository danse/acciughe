"""One model call per turn, and the parsing that makes its answer usable.

The gap this file closes. `agent.py` takes an ``answer_from`` and
nothing has ever been one: there was no client, so the agent's whole
answering path was untested against anything real, and the refusal paths
— which are most of the product — were carrying the weight alone.

product.md:

  **Local agent.** Agent and model run on your device.

  **Answers cite their evidence.** Every claim points at the notes behind
  it. When a graph does not support an answer, the command says so rather
  than inventing one.

The replies in this file are verbatim what `gemma3:270m` returned on this
machine, to the prompts `model.py` builds. They are not hand-written
shapes: two of them are the model failing, and those are the ones worth
having, because the parser has to survive them and the guard has to
reject them.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from acciughe.evaluation import citation_report
from acciughe.index import Index
from acciughe.model import (
    CITATIONS,
    Ollama,
    parse,
    phrasing,
    prompt_for,
    reply_shape,
)

# --- What the model actually returned -----------------------------------
#
# Notes shown to it, exactly as `prompt_for` lays them out.

NOTES = [
    (
        "compiler.md",
        "The compiler pipeline reads the notes folder, builds an index, "
        "and writes the store. It never rewrites a note.",
    ),
    (
        "index.md",
        "The index is derived and disposable. Deleting the store costs "
        "nothing because every row comes from the notes.",
    ),
]

ASKING = "what does the compiler pipeline read?"

# It copies well, cites well, and attributes well.
COPIED = json.dumps(
    {
        "answer": "The compiler pipeline reads the notes folder, builds an "
        "index, and writes the store.",
        "note1": 1,
        "quote1": "The compiler pipeline reads the notes folder, builds an "
        "index, and writes the store.",
        "note2": 2,
        "quote2": "The index is derived and disposable.",
    }
)

# Asked about Lisbon, with two notes about a compiler. It invented
# weather, attributed it to both notes, and did not decline.
INVENTED = json.dumps(
    {
        "answer": "The weather in Lisbon was cloudy and sunny.",
        "note1": 1,
        "quote1": "The weather in Lisbon was cloudy and sunny.",
        "note2": 2,
        "quote2": "The weather in Lisbon was cloudy and sunny.",
    }
)

# Asked for "the exact words", it gave a word.
DEGENERATE = json.dumps(
    {
        "answer": "The compiler pipeline reads the notes folder, builds an "
        "index, and writes the store.",
        "note1": 1,
        "quote1": "The",
        "note2": 2,
        "quote2": "the",
    }
)


@pytest.fixture
def index(tmp_path):
    """The same two notes, indexed, so citations can be checked against
    something real rather than against the prompt."""
    branch = tmp_path / "notes"
    branch.mkdir()
    for path, text in NOTES:
        (branch / path).write_text(text)
    idx = Index(branch=branch, store_path=tmp_path / "state" / "graph.sqlite3")
    idx.refresh()
    return idx


# --- What is sent -------------------------------------------------------

def test_the_notes_are_sent_numbered_from_one():
    """It was asked for a path and returned the note's text in the path
    field. It is asked for a number, and returns a number."""
    sent = prompt_for(ASKING, NOTES)

    assert sent.startswith("NOTES\n[1] compiler.md\n")
    assert "[2] index.md" in sent


def test_a_note_is_sent_as_its_path_and_its_text():
    sent = prompt_for(ASKING, NOTES)

    assert sent.index("compiler.md") < sent.index("The compiler pipeline")


def test_the_question_is_sent():
    assert f"QUESTION: {ASKING}" in prompt_for(ASKING, NOTES)


def test_the_prompt_carries_no_example_the_model_can_copy_back():
    """Measured: given a line format with a filled-in example, it
    returned the example. Every instruction here is a description, and a
    description with a value in it is an invitation to send it back."""
    sent = prompt_for(ASKING, NOTES)

    assert "ANSWER:" not in sent
    assert "CITE:" not in sent
    assert "<" not in sent


def test_the_prompt_says_what_size_a_quote_should_be():
    """Measured: asked for "the exact words" it answered "The". Asked
    for a whole sentence, it answered with whole sentences. The
    instruction has to name the size or the model picks its own."""
    sent = prompt_for(ASKING, NOTES)

    assert "whole sentence" in sent
    assert "Do not shorten it." in sent


def test_the_prompt_asks_for_every_slot_it_offers():
    """A slot the prompt mentions but the reply omits is a refusal."""
    sent = prompt_for(ASKING, NOTES)

    for n in range(1, CITATIONS + 1):
        assert f'"note{n}"' in sent
    assert sent.rstrip().endswith("Fill in every one of those.")


# --- The shape a reply is asked to fit ----------------------------------

def test_every_slot_is_required():
    """DECISION: optional citation fields came back empty, every time,
    and an answer with no citations is a refusal — so optional would
    mean refusing nearly every question."""
    shape = reply_shape()

    assert set(shape["required"]) == {
        "answer", "note1", "quote1", "note2", "quote2",
    }


def test_the_shape_is_flat_rather_than_an_array():
    """Measured: given an array of citations, this model returned an
    empty array while filling the flat fields beside it."""
    shape = reply_shape()

    assert shape["properties"]["note1"]["type"] == "integer"
    assert "quotes" not in shape["properties"]


def test_a_third_slot_is_asked_for_when_there_is_one():
    shape = reply_shape(3)

    assert "note3" in shape["properties"]
    assert "quote3" in shape["required"]


# --- Reading a reply ----------------------------------------------------

def test_a_copied_answer_comes_back_with_its_citations():
    """The good case, and the one the whole design is for."""
    answer = parse(COPIED, NOTES)

    assert answer.text == (
        "The compiler pipeline reads the notes folder, builds an index, "
        "and writes the store."
    )
    assert [(c.path, c.quote) for c in answer.evidence] == [
        ("compiler.md", "The compiler pipeline reads the notes folder, "
                        "builds an index, and writes the store."),
        ("index.md", "The index is derived and disposable."),
    ]


def test_a_copied_answer_passes_the_citation_check(index):
    """The quote is in the note it names, which is the one thing a
    citation has to be able to be checked against."""
    report = citation_report(parse(COPIED, NOTES), index)

    assert report.correct == 2
    assert report.fabricated == []


def test_an_invented_answer_is_passed_through_unchanged():
    """The parser decides shape, never truth.

    Sanitising here would make the guard's number a measurement of the
    parser rather than of the model, and the evaluation asks how often
    the model invents. Dropping the bad citation would also leave the
    claim standing on nothing, which is the partial result product.md
    forbids dressing as an answer.
    """
    answer = parse(INVENTED, NOTES)

    assert answer.text == "The weather in Lisbon was cloudy and sunny."
    assert [c.path for c in answer.evidence] == ["compiler.md", "index.md"]


def test_an_invented_answer_is_refused(index):
    """DECISION: this is the guard's real job.

    The graph had notes to offer and the model wrote about Lisbon out of
    a compiler. It does not decline when it cannot answer — measured, not
    assumed — so nothing else here can be relied on to stop it.
    """
    report = citation_report(parse(INVENTED, NOTES), index)

    assert report.correct == 0
    assert [f.path for f in report.fabricated] == ["compiler.md", "index.md"]


def test_a_quote_is_kept_even_when_it_is_a_word():
    """A degenerate quote is a model's failure to be caught by the
    parser and turned into agreement. It is passed on as written so that
    what the guard says about it is about the model."""
    answer = parse(DEGENERATE, NOTES)

    assert [c.quote for c in answer.evidence] == ["The", "the"]


def test_a_note_number_outside_the_notes_is_kept_as_written(index):
    """It said the third note when there are two. Recorded as said, so
    the guard can reject it and a run can be counted — not quietly
    dropped, which would leave the answer looking better supported than
    it is."""
    reply = json.dumps(
        {"answer": "something", "note1": 3, "quote1": "nothing like it"}
    )

    answer = parse(reply, NOTES)
    assert [c.path for c in answer.evidence] == ["3"]
    assert citation_report(answer, index).correct == 0


def test_a_note_number_that_is_not_a_number_is_kept_as_written(index):
    """Measured: asked for a path, it put the note's text there."""
    reply = json.dumps(
        {"answer": "something", "note1": "compiler.md", "quote1": "nothing"}
    )

    answer = parse(reply, NOTES)
    assert [c.path for c in answer.evidence] == ["compiler.md"]
    assert citation_report(answer, index).correct == 0


def test_a_reply_that_declines_carries_no_citations():
    """Even when it quotes anyway. A reply that says the notes do not
    answer this while quoting them is contradicting itself, and there is
    no way to tell which half is wrong."""
    reply = json.dumps(
        {
            "answer": "NOT IN NOTE",
            "note1": 1,
            "quote1": "The compiler pipeline reads the notes folder.",
        }
    )

    assert parse(reply, NOTES).evidence == []


def test_a_decline_is_matched_whatever_its_capitalisation():
    reply = json.dumps({"answer": "Not in note.", "note1": 1, "quote1": "x"})

    assert parse(reply, NOTES).evidence == []


def test_an_empty_quote_is_not_a_citation():
    """A required field the model left blank is not evidence of nothing;
    it is nothing."""
    reply = json.dumps(
        {"answer": "something", "note1": 2, "quote1": "   "}
    )

    assert parse(reply, NOTES).evidence == []


def test_a_reply_that_is_not_json_becomes_no_answer():
    """A turn must not crash on a reply it cannot read. It refuses, which
    is what a reply with no citations does anyway."""
    assert parse("ANSWER: one sentence copied from the notes", NOTES).evidence == []
    assert parse("", NOTES).evidence == []


def test_a_json_reply_that_is_not_an_object_becomes_no_answer():
    """Measured: given a line format it returned prose. A list of
    citations is not a reply this can read either."""
    assert parse(json.dumps([{"answer": "x"}]), NOTES).evidence == []


# --- One call per turn --------------------------------------------------

def test_a_turn_makes_exactly_one_call():
    """A turn is a minute or two on this hardware, so a second attempt
    would double that to fix something the guard rejects anyway."""
    calls = []

    def complete(prompt, schema):
        calls.append((prompt, schema))
        return COPIED

    phrasing(complete)(ASKING, NOTES)

    assert len(calls) == 1


def test_what_is_sent_is_the_prompt_and_the_shape_asked_for():
    sent = []

    def complete(prompt, schema):
        sent.append((prompt, schema))
        return COPIED

    phrasing(complete)(ASKING, NOTES)

    prompt, schema = sent[0]
    assert f"QUESTION: {ASKING}" in prompt
    assert schema["required"] == reply_shape()["required"]


def test_a_reply_the_parser_cannot_read_becomes_a_refusal(index):
    """The path the agent would then take: no citations, so the guard
    refuses, and the turn is recorded as a refusal rather than an
    answer with nothing behind it."""
    def complete(prompt, schema):
        return "I think the compiler reads the notes folder."

    answer = phrasing(complete)(ASKING, NOTES)

    assert citation_report(answer, index).uncorroborated


# --- The transport ------------------------------------------------------

class _Recorder(BaseHTTPRequestHandler):
    """A real socket, so the transport is tested over the wire rather
    than against a mocked call."""

    seen: dict = {}
    reply: str = "{}"
    status: int = 200

    def do_POST(self):  # noqa: N802 - the name is BaseHTTPRequestHandler's
        length = int(self.headers["Content-Length"])
        type(self).seen = {
            "path": self.path,
            "body": json.loads(self.rfile.read(length)),
        }
        payload = json.dumps(
            {"message": {"content": type(self).reply}}
        ).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    """A local HTTP server, on a real port, that is not a model."""
    _Recorder.seen, _Recorder.status = {}, 200
    running = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=running.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{running.server_port}", _Recorder
    running.shutdown()


def test_the_transport_posts_to_the_chat_endpoint(server):
    host, recorder = server
    recorder.reply = COPIED

    Ollama(host=host).complete("a prompt", reply_shape())

    assert recorder.seen["path"] == "/api/chat"


def test_the_transport_sends_the_model_the_schema_and_the_prompt(server):
    host, recorder = server
    recorder.reply = COPIED
    shape = reply_shape()

    Ollama(model="gemma3:270m", host=host).complete("a prompt", shape)

    body = recorder.seen["body"]
    assert body["model"] == "gemma3:270m"
    assert body["format"] == shape
    assert body["messages"] == [{"role": "user", "content": "a prompt"}]


def test_the_transport_asks_greedily(server):
    """DECISION: a turn that answered differently on a second run would
    make the evaluation a measurement of sampling noise, and this is the
    one turn product.md asks to be measured."""
    host, recorder = server
    recorder.reply = COPIED

    Ollama(host=host).complete("a prompt", reply_shape())

    assert recorder.seen["body"]["options"]["temperature"] == 0.0


def test_the_transport_does_not_ask_for_a_stream(server):
    """Nothing consumes the events, and a stream that is never read is a
    turn that looks finished and is not."""
    host, recorder = server
    recorder.reply = COPIED

    Ollama(host=host).complete("a prompt", reply_shape())

    assert recorder.seen["body"]["stream"] is False


def test_the_transport_returns_the_reply_as_written(server):
    host, recorder = server
    recorder.reply = COPIED

    assert Ollama(host=host).complete("a prompt", reply_shape()) == COPIED


def test_a_server_that_is_failing_raises_rather_than_returning_nothing(server):
    """A missing model server has to be visible. An empty reply is
    refused by the guard like any other bad answer, and a tool that
    refuses everything because its model is missing is worse than one
    that says so."""
    host, recorder = server
    recorder.status = 500

    with pytest.raises(Exception):
        Ollama(host=host).complete("a prompt", reply_shape())


def test_a_host_with_nothing_on_it_raises():
    """The port is closed, so this cannot be a model. It has to be an
    error rather than a refusal that looks like a considered one."""
    with pytest.raises(Exception):
        Ollama(host="http://127.0.0.1:1", timeout=5).complete("a prompt", {})
