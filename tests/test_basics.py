import pytest
from .context import load


def test_loading(capsys, example_fixture):  # pylint: disable=unused-argument
    _ = load()


@pytest.mark.skip()
def test_summarizer():
    sg = load()

    assert sg("Lütfen sadede gel.") == ["Lütfen sadede gel."]
