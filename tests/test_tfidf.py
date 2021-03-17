import pkgutil  # noqa: F401 # pylint: disable=unused-import

import pytest

from .context import Token, tokenizer_context


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_binary_tf():
    with tokenizer_context("bert") as Doc:
        d = Doc(
            "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

        binary = d[0].get_tf("binary")

        assert binary.sum() == 13

        assert binary[Token("Bilişim").id_cs] == 1
        assert binary[Token("sektörü").id_cs] == 1
        assert binary[Token(",").id_cs] == 1
        assert binary[Token("günlük").id_cs] == 1
        assert binary[Token("devrim").id_cs] == 1
        assert binary[Token("##lerin").id_cs] == 1
        assert binary[Token("yaşandığı").id_cs] == 1
        assert binary[Token("ve").id_cs] == 1
        assert binary[Token("bir").id_cs] == 1


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_raw_tf():
    with tokenizer_context("bert") as Doc:
        d = Doc(
            "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

        raw = d[0].get_tf("raw")

        assert raw[Token("Bilişim").id_cs] == 1
        assert raw[Token("sektörü").id_cs] == 1
        assert raw[Token(",").id_cs] == 1
        assert raw[Token("günlük").id_cs] == 1
        assert raw[Token("devrim").id_cs] == 1
        assert raw[Token("##lerin").id_cs] == 1
        assert raw[Token("yaşandığı").id_cs] == 1
        assert raw[Token("ve").id_cs] == 1
        assert raw[Token("bir").id_cs] == 2

    assert raw.sum() == 14


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_raw_tf_without_stopwords():
    with tokenizer_context("bert") as Doc:
        d = Doc(
            "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

        raw = d[0].get_tf("raw", drop_stopwords=True)

        assert raw[Token("Bilişim").id_cs] == 1
        assert raw[Token("sektörü").id_cs] == 1
        assert raw[Token(",").id_cs] == 1
        assert raw[Token("günlük").id_cs] == 1
        assert raw[Token("devrim").id_cs] == 1
        assert raw[Token("##lerin").id_cs] == 1
        assert raw[Token("yaşandığı").id_cs] == 1

        assert raw.sum() == 10


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_raw_tf_without_stopwords_lowercase():
    with tokenizer_context("bert") as Doc:
        d = Doc(
            "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

        raw = d[0].get_tf("raw", drop_stopwords=True, lowercase=True)

        assert raw[Token("bilişim").id] == 1
        assert raw[Token("sektörü").id] == 1
        assert raw[Token(",").id] == 1
        assert raw[Token("günlük").id] == 1
        assert raw[Token("devrim").id] == 1
        assert raw[Token("##lerin").id] == 1
        assert raw[Token("yaşandığı").id] == 1

        assert raw.sum() == 10


@pytest.mark.skipif('pkgutil.find_loader("transformers") is None')
def test_raw_tf_lowercase():
    with tokenizer_context("bert") as Doc:
        d = Doc(
            "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev Bir alan bir olmadı.")

        raw = d[0].get_tf("raw", lowercase=True)

        assert raw[Token("bilişim").id] == 1
        assert raw[Token("sektörü").id] == 1
        assert raw[Token(",").id] == 1
        assert raw[Token("günlük").id] == 1
        assert raw[Token("devrim").id] == 1
        assert raw[Token("##lerin").id] == 1
        assert raw[Token("yaşandığı").id] == 1
        assert raw[Token("ve").id] == 1
        assert raw[Token("bir").id] == 2

        assert raw.sum() == 14
