import pytest

from voice_evals.normalize import cer, normalize, numbers, terms_in, wer


@pytest.mark.parametrize(
    "spoken, written",
    [
        ("What's the list price of T L F twenty-two ten?", "What's the list price of TLF-2210?"),
        ("Tell me about product O R B fifty-two twelve.", "Tell me about product ORB-5212."),
        ("Order six zero seven four four is late.", "Order 60744 is late."),
        ("Has order five eight two one three shipped?", "Has order 58213 shipped?"),
        ("How many Veltra four point five millimeter screws?", "How many Veltra 4.5 mm screws?"),
        ("Is the Tri-Lock stem in size fourteen in stock?", "Is the TriLock stem in size 14 in stock?"),
        ("It lists at one thousand eight hundred forty dollars.", "It lists at $1,840."),
        ("It lists at eighteen hundred and forty dollars.", "It lists at $1,840.00."),
        ("Back in stock on December second.", "Back in stock on December 2nd."),
        ("four packs of Kestrel Bone Cement forty gram are back-ordered", "four packs of Kestrel Bone Cement 40g are backordered"),
        ("Order 58213 shipped and it costs 1,420 USD", "order 58,213 shipped and it costs $1,420"),
        ("diameters from 46 millimeters to 64 millimeters", "diameters from 46 mm to 64mm"),
    ],
)
def test_spoken_and_written_forms_match(spoken, written):
    assert normalize(spoken) == normalize(written)
    assert wer(written, spoken) == 0.0


def test_case_punctuation_and_whitespace_are_ignored():
    assert normalize("  Does the ARCUS knee-tray come in a cementless version?! ") == (
        "does the arcus knee tray come in a cementless version"
    )


def test_numbers_are_extracted_after_normalization():
    assert numbers("Sizes nine to eighteen, $1,840 each, 4.5 mm") == ["9", "18", "1840", "4.5"]
    assert numbers("ORB-5212 is a cup") == []  # SKUs are terms, not numbers


def test_misheard_terms_are_not_hidden_by_the_lexicon():
    assert normalize("try lock") != normalize("TriLock")
    assert normalize("Vectra") != normalize("Veltra")
    assert normalize("TLF-2201") != normalize("TLF-2210")


def test_terms_in_finds_lexicon_terms_in_any_form():
    assert terms_in("the tri lock stem, SKU t l f twenty two ten") == ["TriLock", "TLF-2210"]
    assert terms_in("the try lock stem") == []


def test_wer_counts_substitutions_deletions_and_insertions():
    truth = "is the trilock stem in size fourteen in stock"
    assert wer(truth, "is the trilock stem in size forty in stock") == pytest.approx(1 / 9)
    assert wer(truth, "is the trilock stem in stock") == pytest.approx(3 / 9)
    assert wer(truth, truth + " today") == pytest.approx(1 / 9)
    assert wer("", "") == 0.0
    assert wer("", "anything") == 1.0


def test_cer_is_character_level():
    assert cer("veltra", "vectra") == pytest.approx(1 / 6)


def test_filler_words_are_not_scored():
    assert normalize("What's the list price of, uh, TLF-2210?") == normalize("What's the list price of TLF-2210?")
    assert wer("Hi, um, does the Arcus knee tray come in a cementless version?",
               "Hi, does the Arcus knee tray come in a cementless version?") == 0
    assert normalize("order five eight, uh, two one three") == normalize("order 58213")
    assert "millimeter" in normalize("4.5 mm screws")  # "mm" is a unit, not a filler


def test_v4_rewords_only_its_questions():
    from voice_evals.config import load_golden

    v1, v4 = load_golden("v1"), load_golden("v4")
    changed = [a["utterance_id"] for a, b in zip(v1, v4) if a["gt_transcript"] != b["gt_transcript"]]
    assert changed == ["u_003", "u_005", "u_006", "u_007", "u_009"]
    for a, b in zip(v1, v4):  # expectations never change with the wording
        assert a["expected_tool_calls"] == b["expected_tool_calls"] and a["expected_facts"] == b["expected_facts"]
    # the corrected order number is the one in the expected tool call
    assert "58213" in normalize(v4[8]["gt_transcript"]) and v4[8]["expected_tool_calls"][0]["args"]["order_id"] == "ORD-58213"
