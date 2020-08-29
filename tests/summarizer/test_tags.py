from .context import Rouge1Summarizer
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer
from .context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer


def test_baseline_tags():
    rand = RandomSummarizer()
    pos = PositionSummarizer()
    length = LengthSummarizer()
    band = BandSummarizer()

    assert "baseline" in rand
    assert "baseline" in pos
    assert "baseline" in length
    assert "baseline" in band


def test_cluster_tags():
    km = KMeansSummarizer()
    autokm = AutoKMeansSummarizer()
    decomkm = DecomposedKMeansSummarizer()

    assert "cluster" in km
    assert "cluster" in autokm
    assert "cluster" in decomkm


def test_ss_tags():
    rouge1 = Rouge1Summarizer()

    assert "self-supervised" in rouge1
