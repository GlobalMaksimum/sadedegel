from .context import load


def test_loading(capsys, example_fixture):  # pylint: disable=unused-argument
    _ = load()


def test_summarizer():
    sg = load()

    assert sg("Lütfen sadede gel.") == [["Lütfen", "sade", "##de", "gel", "."]]
