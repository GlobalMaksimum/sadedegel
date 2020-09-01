from .context import get_all_configs, describe_config


def test_all_configs():
    assert isinstance(get_all_configs(), dict)


def test_describe_config_str():
    assert isinstance(describe_config('word_tokenizer'), str)


def test_describe_config_print(capsys):
    describe_config('word_tokenizer', True)
    captured = capsys.readouterr()

    assert 'Change the default word tokenizer used by sadedegel' in captured.out or \
           'Change the default word tokenizer used by sadedegel' in captured.err
