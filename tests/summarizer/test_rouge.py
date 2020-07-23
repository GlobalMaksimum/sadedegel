import pytest
from tests.context import Rouge1Summarizer


@pytest.mark.skip()
def test_rouge_scorer_format():  # test format
    summ = Rouge1Summarizer()
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert len(result) == 3 and len(result[0]) == 2


@pytest.mark.skip()
def test_rouge_scorer_f1():  # test correctness of f1 ordering
    summ = Rouge1Summarizer(metric="f1")
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0


@pytest.mark.skip()
def test_rouge_scorer_recall():
    summ = Rouge1Summarizer(metric="recall")
    result = summ(["aaaaa", "aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0


@pytest.mark.skip()
def test_rouge_summ():
    summ = Rouge1Summarizer(k=2)

    assert summ(["aaaaa", "aaaaaa", "bbb bbb"]) == ["aaaaa", "aaaaaa"]
