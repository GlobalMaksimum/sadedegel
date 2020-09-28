from sadedegel.tokenize import Sentences
from .context import TFIDFSummarizer, Doc
import numpy as np
import pytest


@pytest.mark.parametrize("docs", [["Onu hiç sevmedim", "Bu iş çok zor"]])
@pytest.mark.parametrize("inp_doc", [[Sentences(0, "Bu renk kötü",
                                                Doc("Bu renk kötü. O araç güzel")),
                                      Sentences(1, "O araç güzel",
                                                Doc("Bu renk kötü. O araç güzel"))]])
@pytest.mark.parametrize("result", [np.array([0.23104906, 0.34657359])])
def test_tfidf(docs, inp_doc, result):
    assert TFIDFSummarizer(docs).predict(inp_doc) == pytest.approx(result)
