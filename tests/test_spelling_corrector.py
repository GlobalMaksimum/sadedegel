import pytest
from pytest import raises
from .context import Doc, SpellingCorrector

def test_load_txt_dict():
    sc = SpellingCorrector(dont_use_pickled=True)
    sc._load_dictionary()

def test_fail_dict_load():
    sc = SpellingCorrector(dont_use_pickled=True, dict_path="non-existent path asdf")
    with raises(Exception):
        sc._load_dictionary()

def test_basic():
    sc = SpellingCorrector()
    assert sc.basic("Ali bubanın çiftligi!...") == "Ali babanın çiftliği!..."

def test_compound():
    sc = SpellingCorrector()
    assert sc.compound("Ali bubanın çiftligi!...") == "Ali babanın çiftliği"

def test_doc_basic():
    d = Doc("Ali bubanın çiftligi!...")
    d_fixed = d.get_spell_corrected("basic")

    assert str(d_fixed) == "Ali babanın çiftliği!..."

def test_doc_compound():
    d = Doc("Ali bubanın çiftligi!...")
    d_fixed = d.get_spell_corrected("compound")

    assert str(d_fixed) == "Ali babanın çiftliği"