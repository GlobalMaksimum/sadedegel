from pytest import approx
from tests.context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer, Rouge1Summarizer


def test_firstK_default():
    summarizer = PositionSummarizer(normalize=False)

    assert summarizer.predict([0, 1, 2]) == [2, 1, 0]


def test_firstK_2():
    summarizer = PositionSummarizer()

    assert summarizer.predict([0, 1, 2]) == approx([2 / 3, 1 / 3, 0])


def test_randomK_default():
    summarizer = RandomSummarizer(normalize=False)

    assert summarizer.predict([0, 1, 2]) == approx([2 / 3, 1 / 3, 0])


def test_randomK_2():
    summarizer = RandomSummarizer(normalize=True)

    assert summarizer.predict([0, 1, 2]) == approx([2 / 3, 1 / 3, 0])


def test_rouge_scorer_format():  # test format
    summ = RougeRawScorer()
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert len(result) == 3 and len(result[0]) == 2


def test_rouge_scorer_f1():  # test correctness of f1 ordering
    summ = Rouge1Summarizer(metric="f1")
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0


def test_rouge_scorer_recall():
    summ = Rouge1Summarizer(metric="recall")
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0


def test_rouge_summ():
    summ = Rouge1Summarizer(k=2)

    assert summ(["aaaaa", "aaaaaa", "bbb bbb"]) == ["aaaaa", "aaaaaa"]
