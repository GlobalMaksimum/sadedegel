from .context import Rouge

def test_rogue1_unigrams():
    r1 = Rouge(n=1)

    test_s = "**?,,,~123!=!.       Be gardaş    nabang? Eyidir ya.   "
    expected = ["be", "gardaş", "nabang", "eyidir", "ya"]

    assert r1._get_unigrams(test_s) == expected

def test_rouge1_recall():
    r1 = Rouge(n=1, metric="recall")
    eps = 1e-4

    assert r1("big cat on bed", "My little cat is on bed!") - 3/6 < eps

def test_rouge1_precision():
    r1 = Rouge(n=1, metric="precision")
    eps = 1e-4

    assert r1("big cat on bed", "My little cat is on bed!") - 3/4 < eps

def test_rouge_f1():
    r1 = Rouge(n=1, metric="f1")
    eps = 1e-4

    recall = 3/6
    precision = 3/4
    expected_f1 = (2*recall*precision)/(recall+precision)

    assert r1("big cat on bed", "My little cat is on bed!") - expected_f1 < eps
