import pytest
from gensim.models import Word2Vec
from .context import GCorpus


@pytest.mark.parametrize('corpus, size', [('standard', 4066),
                                          ('extended', 1596797),
                                          ('tokenization', None)])
def test_gensim_corpus(corpus, size):
    if corpus != 'tokenization':
        if corpus == 'extended':
            pytest.skip("Takes longer than 5 min to iterate over all sentences of the corpus.")
        sentences = GCorpus(corpus)
        model = Word2Vec(sentences, iter=1)
        assert model.corpus_count == size
    else:
        with pytest.raises(NotImplementedError, match=r'.*Tokenization Corpus is not yet implemented.*'):
            GCorpus(corpus)
