from itertools import permutations
import pytest
from .context import Text2Doc

preprocess_settings = list(set(permutations([True, True, True, True, False, False, False], 4)))


@pytest.mark.parametrize("settings", preprocess_settings)
def test_text2doc_change(settings):
    vec = Text2Doc(hashtag=settings[0],
                   mention=settings[1],
                   emoji=settings[2],
                   emoticon=settings[3])
    assert vec.Doc.tokenizer.hashtag == settings[0]
    assert vec.Doc.tokenizer.mention == settings[1]
    assert vec.Doc.tokenizer.emoji == settings[2]
    assert vec.Doc.tokenizer.emoticon == settings[3]

    vec = Text2Doc(hashtag=settings[3],
                   mention=settings[2],
                   emoji=settings[1],
                   emoticon=settings[0])
    assert vec.Doc.tokenizer.hashtag == settings[3]
    assert vec.Doc.tokenizer.mention == settings[2]
    assert vec.Doc.tokenizer.emoji == settings[1]
    assert vec.Doc.tokenizer.emoticon == settings[0]
