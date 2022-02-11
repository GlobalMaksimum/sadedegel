import pkgutil  # noqa: F401 # pylint: disable=unused-import


from .context import SupervisedSentenceRanker, RankerOptimizer, Doc
import numpy as np
import pytest
import lightgbm as lgb


famous_quote = ("Merhaba dünya biz dostuz. Barış için geldik. Sizi lazerlerimizle buharlaştırmayacağız."
                " Onun yerine kölemiz olacaksınız.")


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.skipif('pkgutil.find_loader("pandas") is None')
@pytest.mark.skipif('pkgutil.find_loader("optuna") is None')
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("text", [famous_quote])
@pytest.mark.parametrize("vector", ["bert_128k_cased", "bert_32k_cased", "distilbert",  "bert_32k_uncased", "bert_128k_uncased", "electra"])
def test_ranker_init(normalize, text, vector):
    if vector != "bert_128k_cased":
        if vector == "electra":
            with pytest.raises(ValueError, match=r".*Not a valid vectorization for input sequence.*"):
                ranker = SupervisedSentenceRanker(vector_type=vector, debug=True)
        else:
            ranker = SupervisedSentenceRanker(vector_type=vector, debug=True)
            assert ranker.model == f"ranker_{vector}.joblib"

            ranker = SupervisedSentenceRanker(vector_type="bert_128k_cased", debug=True)
            assert ranker.model == f"ranker_bert_128k_cased.joblib"

            ranker = SupervisedSentenceRanker(vector_type="bert_32k_cased")
            assert isinstance(ranker.model, lgb.sklearn.LGBMRanker)
    else:
        ranker = SupervisedSentenceRanker(vector_type=vector)
        assert isinstance(ranker.model, lgb.sklearn.LGBMRanker)


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
@pytest.mark.skipif('pkgutil.find_loader("pandas") is None')
@pytest.mark.skipif('pkgutil.find_loader("optuna") is None')
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("text", [famous_quote])
def test_summary(normalize, text):
    d = Doc(text)
    ranker = SupervisedSentenceRanker(vector_type="bert_128k_cased")
    relevance_scores = ranker.predict(d)
    assert len(relevance_scores) == 4
    if normalize:
        np.sum(relevance_scores) == 1

    for i in range(len(d)):
        assert len(ranker(d, k=i+1)) == i+1
