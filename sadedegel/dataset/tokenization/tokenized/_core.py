import json
import os
from pathlib import Path


_desc = "Tokenization Tokenized Corpus"


def load_corpus(return_iter=True, version='v2', data_home='~/.sadedegel_data'):
    """Load tokenized tokenization corpus.

    :param return_iter: Returns iter[dict] if True List[dict] else, defaults to True
    :type return_iter: bool
    :param version: version of tokenization dataset, defaults to 'v2'
    :type version: str
    :param data_home:

    :returns: Iterator or List of dicts containing tokenized document, doc_id, index and document category if available.
    :rtype: Iter[dict] or List[dict]
    """

    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / version / "tokenized" / "Tokenized.json")) as j:
        tok = json.load(j)

    if return_iter:
        return iter(tok)
    else:
        return tok
