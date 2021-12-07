from math import log

from pytest import approx

from .context import tokenizer_context, Token, Vocabulary


def test_icu_feature():
    v = Vocabulary("icu")

    assert isinstance(v.id_to_feature, dict)
    assert isinstance(v.feature_to_id, dict)
    assert isinstance(v.id_to_feature_cs, dict)
    assert isinstance(v.feature_cs_to_id, dict)


def test_icu_tf():
    with tokenizer_context("icu") as Doc:
        d = Doc("Bir berber bir berbere bire berber gel beraber bir berber dükkanı açalım demiş")

        raw_tf = d.get_tf("raw")

        v = Vocabulary("icu")

        assert raw_tf[v.id_cs("Bir")] == 1
        assert raw_tf[v.id_cs("berber")] == 3
        assert raw_tf[v.id_cs("bir")] == 2
        assert raw_tf[v.id_cs("bire")] == 1
        assert raw_tf[v.id_cs("gel")] == 1
        assert raw_tf[v.id_cs("beraber")] == 1
        assert raw_tf[v.id_cs("dükkanı")] == 1
        assert raw_tf[v.id_cs("açalım")] == 1
        assert raw_tf[v.id_cs("demiş")] == 1

        assert raw_tf.shape == (v.size_cs,)


def test_icu_idf():
    with tokenizer_context("icu") as Doc:
        d = Doc("Bir berber bir berbere bire berber gel beraber bir berber dükkanı açalım demiş")

        smooth_idf = d.get_idf("smooth")

        v = Vocabulary("icu")
        assert Token("Bir").df_cs == 18189
        assert Token("Bir").id_cs == 9705

        assert smooth_idf[v.id_cs("Bir")] == approx(log(v.document_count / (1 + 18189)) + 1)  # 9705
        assert smooth_idf[v.id_cs("berber")] == approx(log(v.document_count / (1 + 32)) + 1)  # 8857
        assert smooth_idf[v.id_cs("bir")] == approx(log(v.document_count / (1 + 36041)) + 1)  # 9705
        assert smooth_idf[v.id_cs("bire")] == approx(log(v.document_count / (1 + 489)) + 1)  # 9764

        assert smooth_idf.shape == (v.size_cs,)


def test_icu_tf_lowercase():
    with tokenizer_context("icu") as Doc:
        d = Doc("Bir berber bir berbere bire berber gel beraber bir berber dükkanı açalım demiş")

        raw_tf = d.get_tf("raw", lowercase=True)

        v = Vocabulary("icu")

        assert raw_tf[v.id("Bir")] == 3
        assert raw_tf[v.id("berber")] == 3
        assert raw_tf[v.id("bir")] == 3
        assert raw_tf[v.id("bire")] == 1

        assert raw_tf.shape == (v.size,)


def test_icu_idf_lowercase():
    with tokenizer_context("icu") as Doc:
        d = Doc("Bir berber bir berbere bire berber gel beraber bir berber dükkanı açalım demiş")

        smooth_idf = d.get_idf("smooth", lowercase=True)

        v = Vocabulary("icu")

        assert Token("Bir").df == 36063
        assert Token("Bir").id == 8965

        assert smooth_idf[v.id("Bir")] == approx(log(v.document_count / (1 + 36063)) + 1)  # 8965
        assert smooth_idf[v.id("berber")] == approx(log(v.document_count / (1 + 52)) + 1)  # 8186
        assert smooth_idf[v.id("bir")] == approx(log(v.document_count / (1 + 36063)) + 1)  # 8965
        assert smooth_idf[v.id("bire")] == approx(log(v.document_count / (1 + 504)) + 1)  # 9001

        assert smooth_idf.shape == (v.size,)
