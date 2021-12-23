from joblib import load as jl_load
from pathlib import Path
from os.path import dirname


def load_model(model_name: str):
    """Loads prebuilt model for inference.

    Parameters
    ----------
    model_name: str
        Model name to be loaded.

    Returns
    -------
    pipeline: object
        sklearn.pipeline.Pipeline object.

    """
    pipeline = jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")

    pipeline.steps[0][1].init()

    return pipeline
