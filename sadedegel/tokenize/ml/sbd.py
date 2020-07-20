from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline
from os.path import dirname
from pathlib import Path
from joblib import dump, load
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
