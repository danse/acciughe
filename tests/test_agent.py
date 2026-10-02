"""A turn: a question, a walk, and an answer or a clean refusal.

The gap this file closes. `Answer` and `Refusal` have existed since
session.py and nothing has ever produced one, so the types were a
promise rather than a behaviour. The seam to the graph is recent too:
`test_walk.py` reaches over a real corpus but reaches nowhere near a
question.

product.md:

  When the graph does not support an answer, the command says so, and
  says which kind of not-knowing it is: absent from the corpus, present
  but unconnected, or thin. It never dresses a partial result as an
  answer, and never answers with hedging where a clean refusal would
  serve. A refusal keeps the nearest notes, marked as not an answer, and
  is inspectable like any other. An answer is given only when the
  evidence supports it.

  Within it, a refusal is a fork rather than a wall: what is absent
  ends the search, what is present but unconnected redirects the
  question, what is thin asks for more.

The load-bearing decision here is that **the refusal is structural**.
Whether the corpus lacks the answer, whether the notes exist but reach
nothing, whether what is there is too little to answer from — all three
are facts about the graph, readable before anything is generated. So
every refusal path below is exercised with no model in the picture at
all, and a refused question costs nothing to ask.

What is left for a model is the part that is genuinely not decidable
here: phrasing an answer, and choosing which spans to quote. A phraser
is passed in rather than stubbed, and the tests supply one that is
deliberately unreliable on purpose.

Two shapes fall out of the walk rather than the other way round. The
notes are offered nearest first, so a bound drops the furthest rather
than an arbitrary one; and a turn keeps both what it accepted and what
the model actually proposed, because a guard that rewrote history would
make citation correctness unmeasurable.
"""

import pytest

from acciughe.agent import MIN_BEYOND, MAX_EVIDENCE, turn
from acciughe.index import Index
from acciughe.session import (
    Answer,
    Citation,
    Next,
    Refusal,
    RefusalKind,
    Session,
)


def corpus(tmp_path, name, **notes):
    branch = tmp_path / name
    branch.mkdir()
    for rel, text in notes.items():
        (branch / f"{rel}.md").write_text(text)
    idx = Index(branch=branch, store_path=tmp_path / "state" / f"{name}.sqlite3")
    idx.refresh()
    return idx


@pytest.fixture
def answerable(tmp_path):
    """A question that lands on one note and walks to two others.

    Twelve notes rather than three, because the branch has to be big
    enough for a word count to mean something: at three notes "about" is
    in two of them, which is a majority and not the whole corpus, so a
    question about weather would match it and look like a question the
    notes partly answer. In a branch of any real size "about" is in
    nearly every note, and the question finds nothing — which is what
    these tests are about.

    The padding shares no word at all with the three notes under test and
    states no link, so it moves the counts and nothing else: the walk
    from `seed` reaches `one` and `two` and stops, and a test can name
    the notes it expects without the padding appearing in a count. Words
    the three notes already share are the ones most likely to be dropped
    by a count, which is why the padding needed adding and why it has to
    avoid exactly the words it is padding for.
    """
    return corpus(
        tmp_path,
        "answerable",
        seed="compiler pipeline corpus. See [[one]] and [[two]].",
        one="branch notes about unrelated gardening",
        two="branch notes about unrelated gardening",
        **{f"pad{n}": text for n, text in enumerate([
            "printer lab duplex jams cartridge",
            "printer lab duplex toner cartridge",
            "allotment butt water moved committee",
            "allotment fence water moved shed",
            "ferry timetable harbour crossing cancelled",
            "ferry harbour crossing tide delayed",
            "beehive frames harvest lavender honey",
            "beehive frames harvest lavender swarm",
            "kayak paddle river rapids practice",
        ])}
    )


@pytest.fixture
def branch_of_real_notes(tmp_path):
    """A branch written the way branches are actually written.

    Every note opens with "the" and several share "about", because those
    are the words a note-taking corpus is full of and a branch that
    lacks them is a fixture rather than a corpus. The subject words
    ("printer", "allotment", "ferry") are in two notes each, which is
    what keeps them distinctive.

    This is what the questions about *absence* need and `answerable` is
    not: a question about Lisbon has to match nothing, and on a
    three-note branch it matched "about" instead, because two notes in
    three was not yet enough for the counting to call "about" a word
    every note has. A language's list calls it one at any size, so the
    refusal a question about Lisbon gets is now the refusal under test
    rather than an accident of how few notes there are.
    """
    notes = {
        "printer": "the printer jams on duplex about the cartridge.",
        "printer2": "the printer wants a new toner about the drum.",
        "allotment": "the allotment butt needs water about now.",
        "allotment2": "the allotment fence wants a coat about now.",
        "ferry": "the ferry timetable changed about the crossing.",
        "ferry2": "the ferry harbour crossing runs about hourly.",
        "hall": "the hall ceiling about the winter wants painting.",
        "shed": "the shed roof about the felt leaked again.",
        "bus": "the bus is late about the roadworks outside.",
        "piano": "the piano wants tuning about twice a year.",
    }
    return corpus(
        tmp_path,
        "real",
        **{name: text for name, text in notes.items()},
    )


@pytest.fixture
def thin(tmp_path):
    """A question that lands on one note and walks to one other."""
    return corpus(
        tmp_path,
        "thin",
        seed="compiler pipeline corpus. See [[one]].",
        one="branch notes about unrelated gardening",
    )


@pytest.fixture
def unconnected(tmp_path):
    """A question that lands on a note which reaches nothing at all."""
    return corpus(
        tmp_path,
        "unconnected",
        lonely="compiler pipeline corpus, all on its own",
        other="branch notes about unrelated gardening",
    )


ASKING = "what does the compiler pipeline read?"


def quoting(*paths):
    """A phraser that answers, and quotes the first line of each note."""
    def phraser(question, evidence):
        quotes = [text.split("\n")[0] for path, text in evidence if path in paths]
        return Answer(
            text="It reads a corpus.",
            evidence=[
                Citation(path=path, quote=quote)
                for path, quote in zip(paths, quotes)
            ],
        )

    return phraser


# --- Finding somewhere to start ----------------------------------------

def test_a_question_lands_on_the_note_whose_words_it_shares(answerable):
    """A question in your own words starts at the note that uses them."""
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert "seed.md" in result.seeds


def test_the_notes_a_walk_reaches_are_offered_nearest_first(answerable):
    """Reaching is what turns a question about one note into a question
    about notes, and the note the question landed on is the nearest of
    them — it is offered too, since excluding the note the walk started
    from would hide what the question was actually about."""
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert [path for path, _ in result.evidence] == [
        "seed.md", "one.md", "two.md",
    ]


def test_the_seeds_are_the_same_however_the_store_is_ordered(tmp_path):
    """A turn that answered differently on a second run would make the
    evaluation a measurement of SQLite's row order."""
    idx = corpus(
        tmp_path,
        "order",
        alpha="compiler pipeline corpus about reading",
        beta="compiler pipeline corpus about writing",
    )

    first = turn(ASKING, idx, answer_from=quoting("alpha.md"))
    second = turn(ASKING, idx, answer_from=quoting("alpha.md"))

    assert first.seeds == second.seeds == ["alpha.md", "beta.md"]


def test_a_note_sharing_more_of_the_question_is_seeded_first(tmp_path):
    """A note that shares three of the question's words is a better place
    to start than one that shares a single common one, and a turn that
    could not tell them apart would be starting at random."""
    idx = corpus(
        tmp_path,
        "rank",
        thin_match="compiler",
        good_match="compiler pipeline corpus",
    )

    assert turn(ASKING, idx, answer_from=quoting()).seeds == [
        "good_match.md",
        "thin_match.md",
    ]


def test_a_word_every_note_has_does_not_seed_a_note(answerable):
    """A stopword is not evidence, and seeding on one would start every
    question from the whole corpus."""
    result = turn("what is it", answerable, answer_from=quoting())

    assert result.seeds == []
    assert result.outcome.kind is RefusalKind.ABSENT


# --- Seeds are ranked by what a shared word says, not how many there are --

def test_one_word_nobody_else_writes_outranks_four_words_everybody_writes(
    tmp_path,
):
    """The count that ranked seeds before this got this backwards.

    The words a question and a note have most often in common are the
    ones with the least to say about either, and on a corpus where
    everybody writes them every candidate shares them equally — so the
    count was ordering the seeds by noise.

    The shape of it: one note names the question's subject and shares
    nothing else, six notes match only on three words that half the
    corpus writes. Under the old count the six won on breadth, three to
    one, and the walk started among notes about something else.

    The frame is three words no language's list knows, because a
    stopword list drops the frame before a seed is ever chosen and there
    would be nothing left to rank. What a corpus really writes in every
    note and no list can know is a habit — a house name, a project's own
    word — and that is the thing this has to be about.
    """
    notes = {"on_topic": "lamination", "on_the_frame": "kappa lambda mu"}
    for n in range(1, 6):
        notes[f"frame_{n}"] = "kappa lambda mu"
    for n in range(1, 6):
        notes[f"quiet_{n}"] = f"aside{n}"
    idx = corpus(tmp_path, "distinguishing", **notes)

    seeds = turn("lamination kappa lambda mu?", idx, answer_from=quoting()).seeds

    assert seeds[0] == "on_topic.md", (
        f"the note about lamination, not the six matching on the frame: {seeds}"
    )
    assert len(seeds) == 7, "and all of them are still seeds: this is a ranking"


def test_a_note_matching_more_distinctive_words_still_wins(tmp_path):
    """Breadth counts, or the weighting would only ever reward one word.

    The distinction from the test above is that those extra shared words
    are worth something. Two rare words in common is better evidence than
    one, and the weighting has to say so or a question naming three parts
    of something would seed on whichever note it mentioned in passing.
    """
    idx = corpus(
        tmp_path,
        "breadth",
        broad="quartz feldspar mica",
        narrow="quartz",
        elsewhere_one="feldspar alone",
        elsewhere_two="mica alone",
    )

    assert turn("quartz feldspar mica?", idx, answer_from=quoting()).seeds == [
        "broad.md",
        "elsewhere_one.md",
        "elsewhere_two.md",
        "narrow.md",
    ], (
        "three rare words beat one. The three that follow tie on a single "
        "rare word each — quartz, feldspar, mica are each in two notes — "
        "so they fall to the documented tie-break on the note's own name, "
        "which is what makes the walk the same however the store is "
        "ordered."
    )


# --- What the model is handed, as opposed to where the walk starts ------

def recording(seen):
    """A phraser that records the notes it was handed, and answers.

    A `Turn` reports what was *kept* of the answer, not what the answer
    was written from, so a test that wants to know which notes the model
    read has to be in the room for the one call that reads them.
    """
    def phraser(question, evidence):
        seen.extend(path for path, _text in evidence)
        path, text = evidence[0]
        return Answer(
            text="It reads a corpus.",
            evidence=[Citation(path=path, quote=text.split("\n")[0])],
        )

    return phraser


def _crowded(tmp_path):
    """A branch where the graph reaches further than the question does.

    Six notes hold "wrong", the one word the question shares with all of
    them and no language's list knows, so all six are seeds and all six
    sit at depth zero — the state the graph on a real branch is nearly
    always in, 41 relations a note there and 39 seeds tied at the same
    depth for one question.

    `zebra` is the only note about lamination and sorts last on the count
    of what it shares, because it shares one word with the question while
    the other six each share a word five notes share. An ordering that
    fell to the note's own name would put the five alphabetically first
    in front of the model and the one note the question was about behind
    them, past the bound of five.
    """
    return corpus(
        tmp_path,
        "crowded",
        zebra="lamination",
        allotment="the allotment went wrong with the hose",
        bus="the bus went wrong with the filter",
        ferry="the ferry went wrong with the knot",
        hall="the hall went wrong with the tables",
        piano="the piano went wrong with the strings",
        shed="the shed went wrong with the brush. See [[apple]] and [[birch]].",
        apple="apple harvest orchard bins",
        birch="birch saplings hedge gaps",
        **{f"pad{n}": text for n, text in enumerate([
            "kayak paddle river rapids",
            "kayak paddle river eddies",
            "beehive frames harvest lavender",
            "beehive frames harvest clover",
        ])}
    )


def test_the_model_reads_the_notes_the_question_is_about(tmp_path):
    """Ranking the seeds was half a fix until this.

    Seeds were ranked by what they shared with the question, and then
    thrown away one line later: `_gather` ordered the evidence by
    `(depth, name)`, every seed is at depth zero, and so the evidence
    handed to the model was the five alphabetically first of the seeds.
    Measured on a 131-note branch, a question about halving was answered
    from five diary notes sharing only *the* and *with*, while the note
    actually about halving — ranked first among the seeds, and not among
    what was read — was left out along with 118 others.

    So the ranking decided the order of a line in the report and nothing
    else. What the model reads is what the answer is written from; if the
    ranking does not reach it, the ranking has not been made.
    """
    seen: list[str] = []

    turn(
        "what went wrong with the lamination?",
        _crowded(tmp_path),
        answer_from=recording(seen),
    )

    assert seen == [
        "zebra.md",
        "allotment.md",
        "bus.md",
        "ferry.md",
        "hall.md",
    ], (
        "the note about lamination first, then the five that share only "
        "'wrong' with the question, in name order because they tie. Under "
        "the old ordering these were the same five without zebra.md, "
        "which fell past the bound."
    )


def test_the_notes_read_run_from_what_the_question_is_about_out_to_what_is_nearest(
    tmp_path,
):
    """All three keys, in the order they are consulted.

    Relevance first: the two seeds share a subject word and so are worth
    something, and they come before every note that shares nothing.
    Depth next, among those notes: `cedar` is two hops out and `apple`
    and `birch` are one, so it reads last of the three. The note's own
    name last of all, which is what makes the order the same however the
    store hands its rows over.

    Depth cannot decide between a seed and a note the question does not
    reach, because there is no such pair: a note holding a word of the
    question *is* a seed, and every seed sits at depth zero by how a
    breadth-first walk starts. So depth ranks what the question says
    nothing about and never overrules what it does.
    """
    idx = corpus(
        tmp_path,
        "ordered",
        alpha="lamination. See [[apple]].",
        cobalt="gasket. See [[birch]].",
        apple="apple orchard bins. See [[cedar]].",
        birch="birch hedge gaps",
        cedar="cedar fence posts",
        pad_one="kayak paddle river rapids",
        pad_two="beehive frames harvest clover",
    )
    seen: list[str] = []

    turn("lamination gasket gone wrong?", idx, answer_from=recording(seen))

    assert seen == [
        "alpha.md",
        "cobalt.md",
        "apple.md",
        "birch.md",
        "cedar.md",
    ], (
        "lamination and gasket are each in one note and so are worth the "
        "most, and they tie on that and fall to the name. None of the "
        "other three shares a word with the question, so they are ordered "
        "on how far the walk reached them: apple and birch at one hop, "
        "cedar at two, with the name breaking the tie between the first "
        "two."
    )


def test_the_notes_read_are_the_same_however_the_store_is_ordered(tmp_path):
    """The sibling of the seed-ordering test, for the evidence.

    A turn that read a different set of notes on a second run would make
    the evaluation a measurement of SQLite's row order, and the bound is
    where that would show: five notes are kept out of however many were
    reached, so a different order means different notes.
    """
    idx = _crowded(tmp_path)
    first: list[str] = []
    second: list[str] = []

    turn(
        "what went wrong with the lamination?",
        idx,
        answer_from=recording(first),
    )
    turn(
        "what went wrong with the lamination?",
        idx,
        answer_from=recording(second),
    )

    assert first == second
    assert len(first) == MAX_EVIDENCE


# --- What the model was handed, counted by how it was reached ------------

def test_the_notes_read_are_split_by_whether_the_question_named_them(answerable):
    """The measurement the evaluation had no way to make before this.

    Every other figure a turn reports is about the model's receipts, and
    none of them can see this: a turn answered from the note the question
    was about and one answered from five notes it was never about both
    quote faithfully, so both score the same. Whether the walk found
    those notes or the question's own words did is invisible from the
    outside, and it is the thing that says whether the graph is load
    bearing.

    `answerable` is the shape: one note holds two of the question's
    words and the other two are its neighbours, holding none. So one note
    came from the question and two came from the links, and a report that
    counted three notes without the split would be counting evidence and
    calling it reach.
    """
    result = turn(ASKING, answerable, answer_from=recording([]))

    assert result.grounding.seeds == 1, "seed.md, which holds the question's words"
    assert result.grounding.through_edges == 2, (
        "one.md and two.md, which the walk reached and the question did not"
    )
    assert result.grounding.read == len(result.evidence), (
        "and the two halves are the whole of it, so a report that sums "
        "them cannot be quietly measuring something narrower"
    )


def test_a_turn_that_read_nothing_reports_no_grounding(branch_of_real_notes):
    """No split exists, so none is reported.

    A question about something unwritten reads no notes at all. Counting
    its absent half as notes that came through the graph would be a walk
    that consulted nothing being recorded as one that reached the graph,
    which is the one direction this figure cannot be wrong in without
    claiming the graph did work it did not do.
    """
    result = turn(
        "what happened in Lisbon?", branch_of_real_notes, answer_from=recording([])
    )

    assert result.evidence == []
    assert result.grounding is None


def _breadth(tmp_path, name, **notes):
    """A branch of ten notes, sized so a rare word can outweigh two
    common ones.

    `distinguishing` weighs a term by the log of how few notes hold it,
    so on ten notes a word in two is worth log 5 and a word in one is
    worth log 10. Two common words together beat one rare word — 4.83
    against 2.30 — while either of them alone loses to it. That gap is
    the whole of the breadth question, and it is a gap in the arithmetic
    rather than in the corpus, which is what makes a fixture for it
    possible at all.
    """
    padding = [
        f"kayak paddle river{n}" for n in range(1, 4)
    ] + [f"beehive frames harvest{n}" for n in range(4, 6)]
    return corpus(tmp_path, name, **notes,
                  **{f"pad{n}": text for n, text in enumerate(padding)})


def test_a_seed_that_wins_on_breadth_and_loses_on_its_best_word_is_a_disagreement(
    tmp_path,
):
    """`seeds_agree` is the breadth question made a measurement.

    The argument for ranking seeds on their single strongest shared term
    is that a note can outrank a better match on the sum: three terms
    worth log 5 each come to more than one term worth log 10, and the
    note holding only the rare word matched more precisely. Whether that
    ever happens on a branch is not something the branch's vocabulary
    says in advance, so a run records it per turn rather than assuming
    either way.

    `broad` shares quartz, feldspar and mica, each in two notes; `deep`
    shares obsidian, in one. The sum ranks broad first at 4.83 against
    2.30, and the strongest single term ranks deep first at log 10
    against log 5. The two disagree, and the turn says so.
    """
    idx = _breadth(
        tmp_path,
        "breadth_wins",
        broad="quartz feldspar mica",
        one="quartz",
        two="feldspar",
        three="mica",
        deep="obsidian",
    )

    result = turn("quartz feldspar mica obsidian?", idx, answer_from=recording([]))

    assert result.seeds[0] == "broad.md", (
        "the sum ranks it first, which is the ranking under test"
    )
    assert result.grounding.seeds_agree is False, (
        "and the strongest-term ranking would have led with deep.md, so "
        "the run records that the sum cost this turn its first note"
    )


def test_a_seed_the_two_rankings_agree_about_is_not_a_disagreement(tmp_path):
    """The common case, and the one that has to be distinguishable from
    the other.

    Every rare word here is in two notes, so the note sharing three of
    them also holds the strongest single term — it wins on both
    rankings. A run where no turn disagrees is a run where the sum has
    cost nothing here, and reporting that as a count is what lets the
    question come off the agenda rather than being argued forever.
    """
    idx = _breadth(
        tmp_path,
        "breadth_agrees",
        broad="quartz feldspar mica",
        one="quartz",
        two="feldspar",
        three="mica",
    )

    result = turn("quartz feldspar mica?", idx, answer_from=recording([]))

    assert result.seeds[0] == "broad.md"
    assert result.grounding.seeds_agree is True


def test_a_note_is_found_by_its_name_even_when_its_words_are_elsewhere(tmp_path):
    """DECISION: a note's name is one of its words.

    A note called ``compiler.md`` that says nothing about compilers is
    still the note meant by a question about compilers, and treating
    the filename as opaque would lose it.
    """
    idx = corpus(tmp_path, "named", compiler="gardening, entirely unrelated")

    result = turn("what did the compiler say?", idx, answer_from=quoting())

    assert result.seeds == ["compiler.md"]


def test_one_shared_word_is_enough_to_start_from(tmp_path):
    """DECISION: a seed is where the walk begins, not evidence for an
    answer, so one shared distinctive word is enough.

    Co-occurrence asks for two, and that is not an inconsistency: it
    derives a relation between two notes and weights it, while a seed
    only says "look here first". Holding a starting point to the standard
    of a derived relation would drop notes the question plainly names.
    """
    idx = corpus(tmp_path, "single", pipeline="notes on something else entirely")

    assert turn("how does the pipeline work?", idx, answer_from=quoting()).seeds == [
        "pipeline.md"
    ]


# --- Absent: what is not in the corpus at all -------------------------

def test_a_question_about_something_unwritten_is_absent(branch_of_real_notes):
    result = turn("what did I conclude about the weather in lisbon?",
                  branch_of_real_notes, answer_from=quoting())

    assert result.outcome.kind is RefusalKind.ABSENT


def test_absent_ends_the_search(branch_of_real_notes):
    """A refusal is a fork rather than a wall, and absence is the wall."""
    result = turn("what did I conclude about the weather in lisbon?",
                  branch_of_real_notes, answer_from=quoting())

    assert result.next_step is Next.END


def test_absent_keeps_no_notes_because_there_are_none_near(branch_of_real_notes):
    """A refusal keeps the nearest notes — and where nothing matched,
    there is nothing near."""
    result = turn("what did I conclude about the weather in lisbon?",
                  branch_of_real_notes, answer_from=quoting())

    assert result.outcome.nearest == []


def test_a_refused_question_never_reaches_the_model(answerable, tmp_path):
    """A refused question costs nothing to ask. On this hardware a turn
    takes a minute or two, and a refusal is the common case — so the
    model must not be consulted to reach one."""
    calls = []

    def phraser(question, evidence):
        calls.append(question)
        return Answer(text="should never be asked", evidence=[])

    turn("what did I conclude about the weather in lisbon?", answerable,
         answer_from=phraser)

    assert calls == []


# --- Unconnected: present, but reaching nothing -----------------------

def test_notes_that_match_but_reach_nothing_are_unconnected(unconnected):
    """Present but unconnected: the notes are there and the question
    lands on one, and that one is an island."""
    result = turn(ASKING, unconnected, answer_from=quoting())

    assert result.outcome.kind is RefusalKind.UNCONNECTED


def test_unconnected_redirects_the_question(unconnected):
    result = turn(ASKING, unconnected, answer_from=quoting())

    assert result.next_step is Next.REDIRECT


def test_unconnected_keeps_the_note_it_landed_on(unconnected):
    """A refusal keeps the nearest notes, and the note the question
    matched is the nearest thing there is."""
    result = turn(ASKING, unconnected, answer_from=quoting())

    assert [n.path for n in result.outcome.nearest] == ["lonely.md"]


def test_the_notes_a_refusal_keeps_are_marked_as_not_being_an_answer(unconnected):
    """They are shown so the reader can see what was found. They are not
    evidence, and cannot be read as evidence."""
    result = turn(ASKING, unconnected, answer_from=quoting())

    assert all(n.is_not_an_answer for n in result.outcome.nearest)


def test_a_redirect_names_the_note_the_walk_stopped_at(unconnected):
    """A refusal that does not say where to go instead is a dead end
    with extra steps, and the note the question matched is the only place
    the walk went."""
    result = turn(ASKING, unconnected, answer_from=quoting())

    assert "lonely.md" in result.outcome.redirect_to


def test_a_redirect_leads_where_the_walk_stopped_and_nowhere_else(tmp_path):
    """Two questions stopping on two islands get two different redirects.
    One fixed phrase would mean the fork is decoration."""
    idx = corpus(
        tmp_path,
        "islands",
        alpha="compiler pipeline corpus, on its own",
        gamma="graph notes about something else",
    )

    on_alpha = turn(ASKING, idx, answer_from=quoting())
    on_gamma = turn("what does the graph note say?", idx, answer_from=quoting())

    assert "alpha.md" in on_alpha.outcome.redirect_to
    assert "gamma.md" in on_gamma.outcome.redirect_to


def test_a_note_that_does_reach_something_is_not_unconnected(answerable):
    """Unconnected is a wall, and it is only the wall when the walk
    found nothing at all. A note that reaches something has not been
    refused for reaching nothing, however little what it reaches is
    worth."""
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert isinstance(result.outcome, Answer)


# --- Thin: there, but too little to answer from -----------------------

def test_a_walk_that_finds_too_little_is_thin(thin):
    """Present and connected, and still not enough. A question that
    spans notes cannot be answered from one."""
    result = turn(ASKING, thin, answer_from=quoting("one.md"))

    assert result.outcome.kind is RefusalKind.THIN


def test_thin_asks_for_more(thin):
    result = turn(ASKING, thin, answer_from=quoting("one.md"))

    assert result.next_step is Next.ASK


def test_thin_keeps_what_it_did_find(thin):
    """The notes are shown so the reader can see how far the walk got.
    They are not an answer, and cannot be read as one."""
    result = turn(ASKING, thin, answer_from=quoting("one.md"))

    assert {n.path for n in result.outcome.nearest} == {"seed.md", "one.md"}


def test_thin_asks_for_the_notes_that_would_have_filled_it(thin):
    """Named after what the walk found, so the reader knows what to write
    toward rather than being told to write more."""
    result = turn(ASKING, thin, answer_from=quoting("one.md"))

    assert "seed.md" in result.outcome.wants


def test_the_two_thinnesses_ask_for_different_things(thin, answerable):
    """Both fork into asking, and they are not the same failure: one
    walk found too little, and one answer failed to say what it stood
    on. Asking for more notes after a citation that does not check out
    would send the reader to write notes, which is not what went wrong.
    """
    walked_thin = turn(ASKING, thin, answer_from=quoting("one.md")).outcome
    answered_thin = turn(ASKING, answerable, answer_from=phrasing_nothing()).outcome

    assert walked_thin.kind is answered_thin.kind is RefusalKind.THIN
    assert walked_thin.wants != answered_thin.wants


def test_the_line_between_thin_and_an_answer_is_a_count(thin, answerable):
    """One note beyond the seed is a lookup. Two is a span. DECISION: the
    threshold is a number rather than a judgement because a judgement
    made by the same code being measured is not a measurement."""
    assert MIN_BEYOND == 2
    assert turn(ASKING, thin, answer_from=quoting("one.md")).outcome.kind \
        is RefusalKind.THIN
    assert isinstance(
        turn(ASKING, answerable, answer_from=quoting("one.md")).outcome, Answer
    )


# --- Answering --------------------------------------------------------

def test_enough_evidence_produces_an_answer(answerable):
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert isinstance(result.outcome, Answer)
    assert result.outcome.text == "It reads a corpus."


def test_an_answer_carries_the_notes_behind_it(answerable):
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert [(c.path, c.quote) for c in result.outcome.evidence] == [
        ("one.md", "branch notes about unrelated gardening")
    ]


def test_answering_says_it_answered(answerable):
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert result.next_step is Next.ANSWERED


# --- The guard: an answer is given only when the evidence supports it --

def test_an_answer_whose_quote_is_not_in_the_note_is_refused(answerable):
    """The command must not invent. A claim with a receipt that does not
    check out is a claim with nothing behind it, whatever it says."""
    def phraser(question, evidence):
        return Answer(
            text="It reads a corpus.",
            evidence=[Citation(path="one.md", quote="a quantum annealer")],
        )

    result = turn(ASKING, answerable, answer_from=phraser)

    assert isinstance(result.outcome, Refusal)


def test_an_answer_citing_a_note_that_does_not_exist_is_refused(answerable):
    def phraser(question, evidence):
        return Answer(
            text="It reads a corpus.",
            evidence=[Citation(path="nowhere.md", quote="anything")],
        )

    result = turn(ASKING, answerable, answer_from=phraser)

    assert isinstance(result.outcome, Refusal)


def test_an_answer_citing_nothing_is_refused(answerable):
    """Every claim points at the notes behind it. A claim with no
    citation has nothing behind it."""

    def phraser(question, evidence):
        return Answer(text="It reads a corpus.", evidence=[])

    result = turn(ASKING, answerable, answer_from=phraser)

    assert isinstance(result.outcome, Refusal)


def test_a_refused_answer_says_so_rather_than_hedging(answerable):
    """Never answers with hedging where a clean refusal would serve. The
    model's own wording — "probably" — is not carried through, because
    carrying it would be dressing a refusal as a soft answer."""
    result = turn(ASKING, answerable, answer_from=phrasing_nothing())

    assert isinstance(result.outcome, Refusal)
    assert result.outcome.explanation
    assert "probably" not in result.outcome.explanation


def phrasing_nothing():
    """A phraser that says something hedged and backs it with nothing."""
    def phraser(question, evidence):
        return Answer(text="It reads a corpus, probably.", evidence=[])

    return phraser


def test_a_refused_answer_keeps_the_notes_it_was_built_from(answerable):
    """The evidence was real even though the claim about it was not, so
    the reader is shown what the question did reach."""
    result = turn(ASKING, answerable, answer_from=phrasing_nothing())

    assert {n.path for n in result.outcome.nearest} == {
        "seed.md", "one.md", "two.md",
    }


def test_what_the_model_proposed_is_kept_alongside_what_was_accepted(answerable):
    """The guard decides what the reader is shown. It must not destroy
    the record of what the model actually said, or citation correctness
    becomes unmeasurable — every answer would be perfect by
    construction."""
    result = turn(ASKING, answerable, answer_from=phrasing_nothing())

    assert result.proposed.text == "It reads a corpus, probably."
    assert result.proposed.evidence == []


def test_the_guard_reports_how_the_citations_checked_out(answerable):
    """So a run of real turns says how often the model invented its
    evidence, which is the number product.md asks to be measured."""
    def phraser(question, evidence):
        return Answer(
            text="It reads a corpus.",
            evidence=[
                Citation(path="one.md", quote="branch notes about unrelated gardening"),
                Citation(path="two.md", quote="a quantum annealer"),
            ],
        )

    result = turn(ASKING, answerable, answer_from=phraser)

    assert result.citations.correct == 1
    assert result.citations.total == 2
    assert [f.path for f in result.citations.fabricated] == ["two.md"]


def test_one_bad_citation_refuses_the_whole_answer(answerable):
    """DECISION: a partly-supported answer is still an answer whose text
    was written against evidence that is not there, and there is no way
    to tell which claims the bad quote was carrying. Dropping the
    citation would leave its claims standing on nothing, which is
    exactly the partial result the spec forbids dressing as an answer.

    The answer is not shown. What is shown instead is the quotation that
    did check out, on its own, with no sentence over it — see
    `test_one_bad_citation_falls_back_to_the_parts_that_checked_out` in
    `test_parts.py`. Nothing here is dressed as an answer either way.
    """
    def phraser(question, evidence):
        return Answer(
            text="It reads a corpus.",
            evidence=[
                Citation(path="one.md", quote="branch notes about unrelated gardening"),
                Citation(path="two.md", quote="a quantum annealer"),
            ],
        )

    result = turn(ASKING, answerable, answer_from=phraser)

    assert not isinstance(result.outcome, Answer)
    assert result.citations.correct == 1


def test_an_answer_whose_citations_all_fail_is_a_refusal(answerable):
    """The rule above, with nothing left to show. A turn that dropped its
    last citation and presented an empty set of parts would be reporting
    a walk that went somewhere and showing nothing from it."""
    def phraser(question, evidence):
        return Answer(
            text="It reads a corpus.",
            evidence=[Citation(path="two.md", quote="a quantum annealer")],
        )

    result = turn(ASKING, answerable, answer_from=phraser)

    assert isinstance(result.outcome, Refusal)


def test_nothing_is_withheld_when_the_walk_finds_less_than_the_bound(answerable):
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert result.withheld == []


def test_the_evidence_a_turn_shows_the_model_is_bounded(tmp_path):
    """An examination says what it left out, which means it has to have
    left something out to say.

    DECISION: the bound is on what the model is shown, not on what the
    walk found. A turn on this hardware is already one to two minutes
    and holds exactly one model call, so the evidence has to fit in that
    call. Bounding the walk instead would face no such limit, and would
    not be a decision anything could be held to.
    """
    assert MAX_EVIDENCE == 5
    beyond = wide(tmp_path)
    assert len(beyond.graph().notes()) > MAX_EVIDENCE
    result = turn(ASKING, beyond, answer_from=quoting("leaf0.md"))

    assert len(result.evidence) == MAX_EVIDENCE


def test_notes_beyond_the_evidence_bound_are_named_not_dropped(tmp_path):
    """The walk found them and the reader is told they were not read. An
    examination that hides what it left out is not inspectable."""
    result = turn(ASKING, wide(tmp_path), answer_from=quoting("leaf0.md"))

    assert result.withheld
    assert set(result.withheld).isdisjoint(
        path for path, _ in result.evidence
    )


def wide(tmp_path):
    """A question whose walk reaches further than the evidence bound."""
    notes = {"seed": "compiler pipeline corpus. See [[one]]."}
    for i in range(MAX_EVIDENCE + 3):
        notes[f"leaf{i}"] = f"branch {i} notes about unrelated gardening"
        notes[f"seed"] += f" [[leaf{i}]]"
    return corpus(tmp_path, "wide", **notes)


# --- The branch is read before it is answered from --------------------

def test_a_note_written_since_the_last_refresh_is_found(answerable, tmp_path):
    """A branch is indexed before it is answered from, and re-read when
    it no longer matches — unasked, and reported rather than silent."""
    (answerable.branch / "three.md").write_text("branch notes about gardening")
    (answerable.branch / "seed.md").write_text(
        "compiler pipeline corpus. See [[one]], [[two]] and [[three]]."
    )

    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert "three.md" in [path for path, _ in result.evidence]


def test_the_work_the_turn_did_is_reported_not_silent(answerable, tmp_path):
    """A refresh happens unasked, and is reported rather than silent."""
    (answerable.branch / "late.md").write_text("a note that arrived late")

    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert result.refresh.added == ["late.md"]


def test_a_turn_that_read_nothing_says_so(answerable):
    """A refresh that changed nothing is as much a report as one that
    did, and reporting only the interesting half would make a quiet turn
    indistinguishable from an unexamined one."""
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert result.refresh.quiet


# --- The conversation -------------------------------------------------

def test_a_turn_records_the_question_and_the_answer(answerable, tmp_path):
    session = Session(path=tmp_path / "sessions", branch=answerable.branch)

    turn(ASKING, answerable, session=session, answer_from=quoting("one.md"))

    roles = [m.role for m in session.messages()]
    assert roles == ["user", "assistant"]
    assert session.messages()[0].content == ASKING


def test_a_refusal_is_recorded_as_a_refusal(unconnected, tmp_path):
    session = Session(path=tmp_path / "sessions", branch=unconnected.branch)

    turn(ASKING, unconnected, session=session, answer_from=quoting())

    said = session.messages()[1]
    assert said.is_answer is False
    assert said.refusal_kind == "unconnected"


def test_a_refused_answer_is_recorded_as_a_refusal(answerable, tmp_path):
    """Not as a hedged answer. The record says what the reader was told."""
    session = Session(path=tmp_path / "sessions", branch=answerable.branch)

    turn(ASKING, answerable, session=session, answer_from=phrasing_nothing())

    assert session.messages()[1].is_answer is False


def test_the_record_keeps_where_the_fork_was_going(unconnected, tmp_path):
    """A redirect is only a fork if it survives being written down. The
    turn that made it is over by the time a conversation is resumed, and
    the note the walk stopped at is the only thing it found."""
    session = Session(path=tmp_path / "sessions", branch=unconnected.branch)

    turn(ASKING, unconnected, session=session, answer_from=quoting())

    resumed = session.reopened().messages()[-1]
    assert resumed.next_step is Next.REDIRECT
    assert "lonely.md" in resumed.redirect_to


def test_a_session_writes_nothing_into_the_branch(answerable, tmp_path):
    """The session is kept outside the branch, since what it records is
    note content."""
    before = sorted(p.name for p in answerable.branch.iterdir())
    session = Session(path=tmp_path / "sessions", branch=answerable.branch)

    turn(ASKING, answerable, session=session, answer_from=quoting("one.md"))

    assert sorted(p.name for p in answerable.branch.iterdir()) == before
    assert session.log_path().exists()
    assert not session.log_path().is_relative_to(answerable.branch)


def test_a_turn_needs_no_session(answerable):
    """The conversation is how a refusal is a fork. A single question is
    still answerable without one."""
    result = turn(ASKING, answerable, answer_from=quoting("one.md"))

    assert isinstance(result.outcome, Answer)
