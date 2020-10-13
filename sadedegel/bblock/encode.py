import numpy as np
from scipy.sparse import csr_matrix
from typing import Union
from .doc import Doc


def encode(document: str, embed='bert', level='sentence') -> Union[np.ndarray, csr_matrix]:
    """High level function for users to encode sentences of given document in desired embedding strategy,
    using SadedeGel building blocks.

    :param document: Input document to be encoded.
    :type document: str
    :param embed: Type of embedding to encode the document's sentences into, defaults to `bert`.
    :type document: str
    :param level: Level of embedding. `sentence` level returns embeddings for each Sentence object in Doc.
    `document` level returns signle embedding for the input document.
    :type level: str

    :return: Encoding
    :rtype: numpy.ndarray, scipy.sparse.csr_matrix
    """

    d = Doc(document)

    if embed == 'bert':
        if level == 'sentence':
            emb = d.bert_embeddings
        elif level == 'document':
            raise NotImplementedError("Doc2Bert is not in current release yet.")
    elif embed == 'tfidf':
        if level == 'sentence':
            emb = d.tfidf_embeddings
        elif level == 'document':
            emb = d.tfidf()
    else:
        raise ValueError("Not a valid embedding type in SadedeGel. Valid values are {\"bert\", \"tfidf\"}")

    return emb
