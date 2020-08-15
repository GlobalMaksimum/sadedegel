from os.path import dirname
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction import DictVectorizer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from joblib import dump, load  # type: ignore

from loguru import logger


def create_model():
    """Creates a new sbd model detector."""
    return Pipeline([('feat', DictVectorizer()), ('dt', RandomForestClassifier())])


def save_model(model, name="sbd.pickle"):
    model_file = (Path(dirname(__file__)) / 'model' / name).absolute()

    dump(model, model_file)


def load_model(name="sbd.pickle"):
    model_file = (Path(dirname(__file__)) / 'model' / name).absolute()
    logger.info(f"Loading sbd model from {model_file}")

    return load(model_file)
