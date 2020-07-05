from .context import FirstK


def test_firstK_default():
    summarizer = FirstK()

    assert summarizer([0, 1, 2]) == [0, 1, 2]


def test_firstK_2():
    summarizer = FirstK(2)

    assert summarizer([0, 1, 2]) == [0, 1]
