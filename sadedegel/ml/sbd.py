from os.path import dirname
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction import DictVectorizer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from joblib import dump, load  # type: ignore

from loguru import logger


class SentenceBoundaryDetector:
    def __init__(self, name: str):
        if name:
            self.model_file = (Path(dirname(__file__)) / 'model' / name).absolute()
        else:
            self.model_file = None
        self.pipeline = None

    def predict(self, spans):
        if self.pipeline is None:
            logger.info(f"Loading sbd model from {self.model_file}")
            self.pipeline = load(self.model_file)

        return self.pipeline.predict(spans)

    def fit(self, features, y):
        self.pipeline.fit(features, y)

    @classmethod
    def from_pipeline(cls, pipeline):
        sbd = SentenceBoundaryDetector(None)
        sbd.pipeline = pipeline

        return sbd


def create_model():
    """Creates a new sbd model detector."""
    pipeline = Pipeline([('feat', DictVectorizer()), ('dt', RandomForestClassifier())])

    return SentenceBoundaryDetector.from_pipeline(pipeline)


def save_model(model: SentenceBoundaryDetector, name="sbd.pickle"):
    model_file = (Path(dirname(__file__)) / 'model' / name).absolute()

    dump(model.pipeline, model_file)


def load_model(name="sbd.pickle"):
    return SentenceBoundaryDetector(name)
