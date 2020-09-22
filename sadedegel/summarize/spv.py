from os.path import dirname
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline  # type: ignore
from joblib import dump, load  # type: ignore

from loguru import logger


def get_lgbm():

    params = {'n_estimators': 500,
              'max_depth': 6,
              'n_jobs': -1,
              'subsample': 0.7}

    model = LGBMRegressor(**params)

    logger.info(f'Initialized LightGBM Model with: {params}')

    return 'lgb', model

def get_rf():

    params = {'n_estimators': 500,
              'criterion': 'rmse',
              'max_samples': 0.7,
              'max_features': 0.7,
              'n_jobs': -1,
              'max_depth': 6}

    model = RandomForestRegressor(**params)

    logger.info(f'Initialized Random Forest Model with: {params}')

    return 'rf', model


def create_model(model_type='rf'):

    if model_type == 'rf':
        name, model = get_rf()
    elif model_type == 'lgbm':
        name, model = get_lgbm()
    else:
        raise NotImplementedError('Model type not implemented yet.')

    return Pipeline([(name, model)])


def save_model(model, name):
    model_file = (Path(dirname(__file__)) / 'model' / f'{name}.pickle').absolute()

    dump(model, model_file)


def load_model(name):
    model_file = (Path(dirname(__file__)) / 'model' / f'{name}.pickle').absolute()
    logger.info(f"Loading supervised model from {model_file}")

    return load(model_file)


