from .context import FirstK, RandomK, RougeRawScorer,RougeSummarizer


def test_firstK_default():
    summarizer = FirstK()

    assert summarizer([0, 1, 2]) == [0, 1, 2]


def test_firstK_2():
    summarizer = FirstK(2)

    assert summarizer([0, 1, 2]) == [0, 1]


def test_randomK_default():
    summarizer = RandomK()

    assert set(summarizer([0, 1, 2])) == set([0, 1, 2])


def test_randomK_2():
    summarizer = RandomK(2)

    assert set(summarizer([0, 1, 2])) in [{0, 1}, {0, 2}, {1, 2}]


def test_rouge_scorer_format(): # test format
    summ = RougeRawScorer()
    result = summ(["aaaaa","aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert len(result) == 3 and len(result[0]) == 2

def test_rouge_scorer_f1(): # test correctness of f1 ordering
    summ = RougeRawScorer(metric="f1")
    result = summ(["aaaaa","aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0

def test_rouge_scorer_recall():
    summ = RougeRawScorer(metric="recall")
    result = summ(["aaaaa","aaaaaa", "bbb bbb"])
    ## bbb bbb should be the last sentence with f1 == 0
    assert result[-1][0] == 2 and result[-1][1] == 0

def test_rouge_summ():
    summ = RougeSummarizer(k=2)

    assert summ(["aaaaa","aaaaaa", "bbb bbb"]) == ["aaaaa","aaaaaa"]
