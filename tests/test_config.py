from pytest import raises
import pytest
from .context import get_all_configs, describe_config, set_config, Doc, config_context


def test_default():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert Doc.config['tf']['method'] == "log_norm"
    assert d.builder.config['tf']['method'] == "log_norm"


def test_config_context():
    for tf in ["binary", "raw"]:
        with config_context(tf__method=tf) as Doc_with_context:
            d = Doc_with_context("Ali topu tut. Ömer ılık süt iç.")

            assert Doc_with_context.config['tf']['method'] == tf
            assert d.builder.config['tf']['method'] == tf
            assert all(s.tf_method == tf for s in d)


@pytest.mark.skip()
def test_all_configs():
    assert isinstance(get_all_configs(), dict)


@pytest.mark.skip()
def test_describe_config_str():
    assert isinstance(describe_config('word_tokenizer'), str)


@pytest.mark.skip()
def test_describe_config_print(capsys):
    describe_config('word_tokenizer', True)
    captured = capsys.readouterr()

    assert 'word_tokenizer is used to split sentences into words.' in captured.out or \
           'word_tokenizer is used to split sentences into words.' in captured.err


@pytest.mark.skip()
def test_tf_setting():
    with raises(Exception, match=r".*is not a valid value.*"):
        set_config('tf', 'double_norm_k')
