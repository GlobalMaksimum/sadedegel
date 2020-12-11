from .context import Doc
from .context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer, Rouge1Summarizer
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, TextRank, TFIDFSummarizer
from pytest import warns
import pytest
import itertools

famous_quote = "Merhaba dünya. Barış için geldik. Sizi lazerlerimizle eritmeyeceğiz."

ks = [0, 1, 2, 3, 4]
summarizers = [RandomSummarizer, PositionSummarizer,
               LengthSummarizer, BandSummarizer, Rouge1Summarizer,
               KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer,
               TextRank, TFIDFSummarizer]


@pytest.mark.parametrize("k, summarizer", itertools.product(ks, summarizers))
def test_k(k, summarizer):
    d = Doc(famous_quote)
    if k > len(d):
        with warns(UserWarning, match=r"State a summary size"):
            summary = summarizer()(d, k=k)
    elif k == 0:
        assert not LengthSummarizer()(d, k=k)
    else:
        assert len(LengthSummarizer()(d, k=k)) == k




