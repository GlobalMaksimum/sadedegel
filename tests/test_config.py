from pytest import raises
from .context import get_all_configs, describe_config, set_config


def test_all_configs():
    assert isinstance(get_all_configs(), dict)


def test_describe_config_str():
    assert isinstance(describe_config('word_tokenizer'), str)


def test_describe_config_print(capsys):
    describe_config('word_tokenizer', True)
    captured = capsys.readouterr()

    assert 'word_tokenizer is used to split sentences into words.' in captured.out or \
           'word_tokenizer is used to split sentences into words.' in captured.err


def test_tf_setting():
    with raises(Exception, match=r".*is not a valid value.*"):
        set_config('tf', 'double_norm_k')
