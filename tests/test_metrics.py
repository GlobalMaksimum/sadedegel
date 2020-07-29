from pytest import approx
from .context import rouge1_score


def test_rouge1_recall():
    assert rouge1_score(["big", "cat", "on", "bed"], ["My", "little", "cat", "is", "on", "bed", "!"],
                        metric="recall") == approx(3 / 4)


def test_rouge1_precision():
    assert rouge1_score(["big", "cat", "on", "bed"], ["My", "little", "cat", "is", "on", "bed", "!"],
                        metric="precision") == approx(3 / 7)


def test_rouge_f1():
    recall = 3 / 4
    precision = 3 / 7
    expected_f1 = (2 * recall * precision) / (recall + precision)

    assert rouge1_score(["big", "cat", "on", "bed"], ["My", "little", "cat", "is", "on", "bed", "!"],
                        metric="f1") == approx(expected_f1)


# Test Case: https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/
def test_rouge1_recall_2():
    assert rouge1_score(["the", "cat", "was", "under", "the", "bed"],
                        ["the", "cat", "was", "found", "under", "the", "bed"],

                        metric="recall") == 1.


def test_rouge1_precision_2():
    assert rouge1_score(["the", "cat", "was", "under", "the", "bed"],
                        ["the", "cat", "was", "found", "under", "the", "bed"],
                        metric="precision") == approx(6 / 7)


def test_rouge_f1_2():
    recall = 1.
    precision = 6 / 7
    expected_f1 = (2 * recall * precision) / (recall + precision)

    assert rouge1_score(["the", "cat", "was", "under", "the", "bed"],
                        ["the", "cat", "was", "found", "under", "the", "bed"],
                        metric="f1") == approx(expected_f1)


def test_rouge1_empty_ycand():
    assert rouge1_score([], ["test"], metric="f1") == 0.0


def test_rouge1_empty_yref():
    assert rouge1_score(["test"], [], metric="f1") == 0.0


def test_rouge1_empty_all():
    assert rouge1_score([], [], metric="f1") == 0.0


def test_rouge1_no_common():
    assert rouge1_score(["the", "cat"], ["a", "dog"], metric="f1") == 0.0
