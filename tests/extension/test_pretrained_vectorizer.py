from .context import PreTrainedVectorizer, load_tweet_sentiment_train
from sklearn.pipeline import Pipeline

mini_corpus = ['Sabah bir tweet', 'Öğlen bir başka tweet', 'Akşam bir tweet', '...ve gece son bir tweet']


def test_vectorizer_pipeline():
    vecpipe = Pipeline([("bert_doc_embeddings", PreTrainedVectorizer(model="bert_32k_cased", do_sents=False, progress_tracking=False))])
    embs = vecpipe.fit_transform(mini_corpus)

    assert embs.shape[0] == len(mini_corpus)
