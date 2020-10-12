import json
import os
from pathlib import Path


def load_corpus(return_iter=True, version='v2', data_home='~/.sadedegel_data'):
    """Load raw tokenization corpus.

    :param return_iter: Returns iter[dict] if True List[dict] else, defaults to True
    :type return_iter: bool
    :param version: version of tokenization dataset, defaults to 'v2'
    :type version: str
    :param data_home:

    :returns: Iterator or List of dicts containing raw document, doc_id, index and document category if available.
    :rtype: Iter[dict] or List[dict]
    """

    with open(str(Path(os.path.expanduser(data_home)) / 'tokenization' / version / "raw" / "Raw.json")) as j:
        raw = json.load(j)

    if return_iter:
        return iter(raw)
    else:
        return raw
