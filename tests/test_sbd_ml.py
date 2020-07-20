from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import PassiveAggressiveClassifier as PA
from sklearn.pipeline import Pipeline
from .context import load_raw_corpus, load_sentence_corpus, Doc, flatten, is_eos, create_model, load_model, save_model


def test_span_feature_hashed():
    raw_corpus = load_raw_corpus()

    features = flatten([[span.span_features() for span in Doc(raw).spans] for raw in raw_corpus])

    hasher = FeatureHasher()

    X = hasher.transform(features)

    assert X.shape[1] == hasher.n_features


def test_model_train_explicit():
    raw_corpus = load_raw_corpus(False)
    sent_corpus = load_sentence_corpus(False)

    features = flatten([[span.span_features() for span in Doc(raw).spans] for raw in raw_corpus])
    y = flatten(
        [[is_eos(span, sent['sentences']) for span in Doc(raw).spans] for raw, sent in zip(raw_corpus, sent_corpus)])

    assert len(features) == len(y)

    pipeline = Pipeline([('hasher', FeatureHasher()), ('pa', PA())])

    pipeline.fit(features, y)


def test_model_train_implicit():
    raw_corpus = load_raw_corpus(False)
    sent_corpus = load_sentence_corpus(False)

    features = flatten([[span.span_features() for span in Doc(raw).spans] for raw in raw_corpus])
    y = flatten(
        [[is_eos(span, sent['sentences']) for span in Doc(raw).spans] for raw, sent in zip(raw_corpus, sent_corpus)])

    assert len(features) == len(y)

    sbd_model = create_model()

    sbd_model.fit(features, y)

    save_model(sbd_model, "sbd.test.pickle")

    del sbd_model

    _ = load_model("sbd.test.pickle")
