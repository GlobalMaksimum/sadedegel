import pkgutil  # noqa: F401 # pylint: disable=unused-import

from .context import Doc
from .context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer, Rouge1Summarizer
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer, TextRank, TFIDFSummarizer
from pytest import warns
import pytest
import itertools

famous_quote = "Merhaba dünya. Barış için geldik. Sizi lazerlerimizle eritmeyeceğiz."

ks = [0, 1, 2, 3, 4]
summarizers = [RandomSummarizer(), PositionSummarizer(),
               LengthSummarizer(), Rouge1Summarizer(),
               KMeansSummarizer(), AutoKMeansSummarizer(), DecomposedKMeansSummarizer(),
               TextRank(), TFIDFSummarizer()]


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.parametrize("k, summarizer", itertools.product(ks, summarizers))
def test_cardinality(k, summarizer):
    d = Doc(famous_quote)

    if k > len(d):
        with warns(UserWarning, match=r"is greater then the number of sentences"):
            summary = summarizer(d, k)
    else:
        summary = summarizer(d, k)

    if k > len(d):
        assert len(summary) == len(d)
    elif k == 0:
        assert len(summary) == 0
    else:
        assert len(summary) == k
