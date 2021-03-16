# flake8: noqa
from pathlib import Path  # pylint: disable=unused-import
from os.path import expanduser  # pylint: disable=unused-import
import pytest
from .context import tweet_sentiment


@pytest.mark.skip()
@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_cv():
    tweet_sentiment.cv(k=5)


@pytest.mark.skip()
@pytest.mark.skipif('not Path(expanduser("~/.sadedegel_data")).exists()')
def test_build():
    tweet_sentiment.build()
