from .context import evaluate, filter_summary, SimpleTokenizer, BertTokenizer
from click.testing import CliRunner
import pytest


@pytest.mark.parametrize("tag, summarizers", [("extractive", ['Random Summarizer', 'FirstK Summarizer',
                                                              'LastK Summarizer', 'Rouge1 Summarizer (f1)',
                                                              'Rouge1 Summarizer (precision)',
                                                              'Rouge1 Summarizer (recall)',
                                                              'Length Summarizer (char)',
                                                              'Length Summarizer (token)',
                                                              'KMeans Summarizer',
                                                              'AutoKMeans Summarizer',
                                                              'DecomposedKMeans Summarizer']),
                                              ("baseline", ['Random Summarizer', 'FirstK Summarizer',
                                                            'LastK Summarizer',
                                                            'Length Summarizer (char)',
                                                            'Length Summarizer (token)']),
                                              ("self-supervised", ['Rouge1 Summarizer (f1)',
                                                                   'Rouge1 Summarizer (precision)',
                                                                   'Rouge1 Summarizer (recall)']),
                                              ("ml",  ['Rouge1 Summarizer (f1)',
                                                       'Rouge1 Summarizer (precision)',
                                                       'Rouge1 Summarizer (recall)',
                                                       'KMeans Summarizer',
                                                       'AutoKMeans Summarizer',
                                                       'DecomposedKMeans Summarizer'])])
def test_tag_filter(tag, summarizers):

    summ = filter_summary(tag)
    summ = [s[0] for s in summ]
    assert summ == summarizers


@pytest.mark.parametrize("tokenizer, true_shape",
                         [(BertTokenizer.__name__, "(5, 4)"), (SimpleTokenizer.__name__, "(5, 4)")])
def test_eval_baseline(tokenizer, true_shape):
    runner = CliRunner()
    result = runner.invoke(evaluate, ['--debug', 'True', '--tag', 'baseline', '-wt', f'{tokenizer}'])
    assert result.output[-10:].rsplit('\n')[-2] == true_shape


@pytest.mark.parametrize("tokenizer, true_shape",
                         [(BertTokenizer.__name__, "(3, 4)"), (SimpleTokenizer.__name__, "(3, 4)")])
def test_eval_selfsupervised(tokenizer, true_shape):
    runner = CliRunner()
    result = runner.invoke(evaluate, ['--debug', 'True', '--tag', 'self-supervised', '-wt', f'{tokenizer}'])
    assert result.output[-10:].rsplit('\n')[-2] == true_shape


@pytest.mark.skip(reason="Takes long to test. "
                         "User can decide not to skip if there are any changes in eval script or ML based summarizers.")
@pytest.mark.parametrize("tokenizer, true_shape",
                         [(BertTokenizer.__name__, "(6, 4)"), (SimpleTokenizer.__name__, "(6, 4)")])
def test_eval_ml(tokenizer, true_shape):
    runner = CliRunner()
    result = runner.invoke(evaluate, ['--debug', 'True', '--tag', 'ml', '-wt', f'{tokenizer}'])
    assert result.output[-10:].rsplit('\n')[-2] == true_shape
