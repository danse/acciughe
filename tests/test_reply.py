"""Which of the two things a reply offers gets shown.

`test_model.py` pins the transport and the parsing. This file pins the
judgement in between: a reply carries a sentence *and* the quoted spans
it was drawn from, and this decides which of them the reader is given.

The measured behaviour that makes the judgement necessary. Asked a
question it can answer, `gemma3:270m` copies the sentence out of the
note. Asked one it cannot — asked about the weather with two notes about
a compiler — it wrote "The weather in Lisbon was cloudy and sunny",
attributed it to both notes, and did not decline. The sentence is the
only part of that reply no check reaches: the guard can tell a quote is
not in the note it names, and it can do nothing at all with a sentence.

So the sentence is shown when the notes contain it and set aside when
they do not, and the spans are shown either way.

**One call, both outcomes.** A turn is one to two minutes on two SSE4.2
cores, so a fallback that asked again would be spending a second call on
exactly the turns that have already failed once. Both answers come out of
the reply that was already paid for.
"""

import json

from acciughe.model import parse, phrasing, prompt_for, reply_shape
from acciughe.session import Answer, Components

from test_model import ASKING, COPIED, NOTES

# The same good answer, with the model having said something the notes do
# not contain instead of copying.
INVENTED_SENTENCE = json.dumps(
    {
        "answer": "The pipeline reads, indexes and caches, which is the "
                  "usual arrangement for this kind of tool.",
        "note1": 1,
        "quote1": "The compiler pipeline reads the notes folder, builds an "
                  "index, and writes the store.",
        "note2": 2,
        "quote2": "The index is derived and disposable.",
    }
)


def phrasing_over(reply):
    calls = []

    def complete(prompt, schema):
        calls.append((prompt, schema))
        return reply

    return phrasing(complete), calls


# --- The model gets first refusal ---------------------------------------

def test_a_sentence_the_notes_contain_is_shown_as_an_answer():
    answer_from, _ = phrasing_over(COPIED)

    result = answer_from(ASKING, NOTES)

    assert isinstance(result, Answer)
    assert result.text.startswith("The compiler pipeline reads")


def test_a_sentence_in_no_note_is_not_shown():
    """product.md: "It never dresses a partial result as an answer." A
    sentence that is not in the notes is the model's own, and the reader
    is not shown it as though the notes had said it."""
    answer_from, _ = phrasing_over(INVENTED_SENTENCE)

    result = answer_from(ASKING, NOTES)

    assert not isinstance(result, Answer)
    assert isinstance(result, Components)


def test_the_parts_shown_are_the_quotes_the_reply_already_carried():
    """The walk's work is not thrown away with the sentence. Both spans
    were verbatim and attributable, and they are the only parts of that
    reply a reader can check."""
    answer_from, _ = phrasing_over(INVENTED_SENTENCE)

    result = answer_from(ASKING, NOTES)

    assert [(c.path, c.quote) for c in result.evidence] == [
        ("compiler.md", "The compiler pipeline reads the notes folder, "
                        "builds an index, and writes the store."),
        ("index.md", "The index is derived and disposable."),
    ]


def test_a_quote_is_never_repaired_on_the_way_to_being_shown():
    """The guard is what rejects a quote that is not in its note, and it
    can only do that if the quote reaches it as written. Cleaning it up
    here would make the guard's number a measurement of this function."""
    reply = json.dumps(
        {
            "answer": "Something in no note at all.",
            "note1": 1,
            "quote1": "The weather in Lisbon was cloudy and sunny.",
        }
    )
    answer_from, _ = phrasing_over(reply)

    result = answer_from(ASKING, NOTES)

    assert [c.quote for c in result.evidence] == [
        "The weather in Lisbon was cloudy and sunny."
    ]


# --- One call, whichever way it goes ------------------------------------

def test_a_sentence_set_aside_costs_no_second_call():
    answer_from, calls = phrasing_over(INVENTED_SENTENCE)

    answer_from(ASKING, NOTES)

    assert len(calls) == 1


def test_both_outcomes_come_from_the_same_reply():
    """The point of the design. One call, and which of the two things in
    the reply to show is decided from it rather than by asking again."""
    shown = [
        phrasing_over(reply)[0](ASKING, NOTES).__class__
        for reply in (COPIED, INVENTED_SENTENCE)
    ]

    assert shown == [Answer, Components]


# --- What counts as a copy ----------------------------------------------

def _shown(reply_text, evidence=NOTES):
    return phrasing_over(reply_text)[0](ASKING, evidence)


def test_a_sentence_copied_onto_one_line_is_still_a_copy():
    """A line break in a note is not a difference in what it says. This is
    the same forgiveness the citation check already applies, for the same
    reason: the author did not think about it."""
    reply = json.dumps(
        {
            "answer": "The index is derived and disposable. Deleting the "
                      "store costs nothing because every row comes from "
                      "the notes.",
            "note1": 2,
            "quote1": "The index is derived and disposable.",
        }
    )

    assert isinstance(_shown(reply), Answer)


def test_a_sentence_with_a_trailing_full_stop_is_still_a_copy():
    reply = json.dumps(
        {
            "answer": "The compiler pipeline reads the notes folder.",
            "note1": 1,
            "quote1": "The compiler pipeline reads the notes folder",
        }
    )

    assert isinstance(_shown(reply), Answer)


def test_a_sentence_assembled_from_a_note_it_did_not_quote_is_still_a_copy():
    """DECISION: a sentence is set aside when it appears in *no* note
    shown, not when it is absent from the ones it cited. The model
    legitimately draws on a note it has nothing to quote from, and
    anything finer means judging whether the sentence follows from the
    spans — a second model call, on a turn that is already a minute or
    two."""
    reply = json.dumps(
        {
            "answer": "branch notes about unrelated gardening",
            "note1": 1,
            "quote1": "branch notes about unrelated gardening",
        }
    )
    evidence = [("one.md", "branch notes about unrelated gardening")]

    assert isinstance(_shown(reply, evidence), Answer)


def test_a_sentence_in_different_words_from_every_note_is_set_aside():
    reply = json.dumps(
        {
            "answer": "The corpus is scanned before it is queried, so a "
                      "stale read cannot be answered from.",
            "note1": 1,
            "quote1": "The compiler pipeline reads the notes folder, "
                      "builds an index, and writes the store.",
        }
    )

    assert isinstance(_shown(reply), Components)


def test_an_empty_sentence_is_not_a_copy():
    """Nothing said is not a sentence copied from a note, and treating the
    empty string as a substring of everything would claim otherwise."""
    reply = json.dumps(
        {
            "answer": "",
            "note1": 1,
            "quote1": "The compiler pipeline reads the notes folder.",
        }
    )

    assert isinstance(_shown(reply), Components)


def test_a_sentence_in_no_note_shows_the_parts_even_with_no_quotes():
    """Both halves of a reply can fail. What is left is nothing, and that
    is the guard's to refuse rather than this function's to dress up."""
    reply = json.dumps({"answer": "Something nobody wrote."})

    assert isinstance(_shown(reply), Components)
    assert parse(reply, NOTES).evidence == []


def test_a_shorter_quote_in_the_same_words_does_not_make_a_copy():
    """The check is on the sentence, not on whether some fragment of it
    was quoted. Quoting "The index" and writing a paragraph is still a
    paragraph."""
    reply = json.dumps(
        {
            "answer": "The index is derived, disposable, and rebuilt from "
                      "the notes every single time the branch moves.",
            "note1": 2,
            "quote1": "The index",
        }
    )

    assert isinstance(_shown(reply), Components)


# --- What is sent is unchanged ------------------------------------------

def test_the_prompt_still_asks_for_a_sentence():
    """The model is not told to stop concluding. It is asked for an answer
    first refusal, and overruled only when the notes do not contain what
    it wrote — a judgement made here, where the notes are in hand."""
    sent = prompt_for(ASKING, NOTES)

    assert "Set \"answer\" to one whole sentence copied from the notes." in sent


def test_the_shape_asked_for_is_unchanged():
    assert "answer" in reply_shape()["required"]
