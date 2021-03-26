from joblib import load as jl_load
from pathlib import Path
from os.path import dirname


def load_model(model_name: str):
    pipeline = jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")

    pipeline.steps[0][1].init()

    return pipeline
