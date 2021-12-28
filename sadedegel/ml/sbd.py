from os.path import dirname
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction import DictVectorizer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from joblib import dump, load  # type: ignore

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DictionaryType, StringType, Int64TensorType, FloatTensorType, StringTensorType
import onnxruntime as rt

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

            if ".pickle" in str(self.model_file):
                self.pipeline = load(self.model_file)
            elif ".onnx" in str(self.model_file):
                self.pipeline = rt.InferenceSession("sadedegel/ml/model/sbd.onnx", providers=rt.get_available_providers())

        if ".pickle" in str(self.model_file):
            return self.pipeline.predict(spans)
        elif ".onnx" in str(self.model_file):
            inp, out = self.pipeline.get_inputs()[0], self.pipeline.get_outputs()[0]
            preds = [self.pipeline.run([out.name], {inp.name: feats})[0][0] for feats in spans if feats]
            return preds

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


def save_model(pipeline: SentenceBoundaryDetector, name="sbd.onnx"):
    model_file = (Path(dirname(__file__)) / 'model' / name).absolute()

    logger.info("Converting sklearn pipeline to ONNX format")

    initial_type = [('boolean_input', DictionaryType(StringType(), FloatTensorType()))]
    onx = convert_sklearn(pipeline.pipeline, initial_types=initial_type)
    with open(model_file, "wb") as f:
        f.write(onx.SerializeToString())

    #dump(pipeline.pipeline, model_file)


def load_model(name="sbd.onnx"):
    return SentenceBoundaryDetector(name)
