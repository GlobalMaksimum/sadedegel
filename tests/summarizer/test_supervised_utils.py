from .context import create_model, save_model, load_model
import pytest
from pytest import raises


@pytest.mark.parametrize('model_type, mdl', [('rf', 'RandomForestRegressor'),
                                             ('lgbm', 'LGBMRegressor'),
                                             ('nn', 'no model')])
def test_create_model(model_type, mdl):
    if model_type == 'rf' or model_type == 'lgbm':
        assert create_model(model_type)._final_estimator.__str__().split('(', 1)[0] == mdl
    else:
        with raises(NotImplementedError, match=r'.*not implemented yet.'):
            create_model(model_type)


@pytest.mark.parametrize('model_type, mdl', [('rf', 'RandomForestRegressor'),
                                             ('lgbm', 'LGBMRegressor')])
def test_save_load(model_type, mdl):
    model = create_model(model_type)

    save_model(model, model_type)

    del model

    loaded = load_model(model_type)

    assert loaded._final_estimator.__str__().split('(', 1)[0] == mdl
