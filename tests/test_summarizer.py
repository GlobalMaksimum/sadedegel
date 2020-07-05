from .context import FirstK, RandomK


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
