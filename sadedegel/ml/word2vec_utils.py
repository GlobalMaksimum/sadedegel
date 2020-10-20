from sadedegel.dataset import load_raw_corpus
from sadedegel.dataset.extended import load_extended_raw_corpus
from sadedegel import Doc
from sadedegel.bblock.word_tokenizer_helper import puncts
from sadedegel.bblock.util import tr_lower
from sadedegel.config import tokenizer_context

from tqdm import tqdm


class GCorpus(object):
    def __init__(self, sadedegel_corpus='standard', tokenizer='simple'):
        self._corpus_type = sadedegel_corpus
        self._corpus = None
        self.toker = tokenizer

        if self._corpus_type == 'standard':
            self._corpus = load_raw_corpus()
            self.total = 98
        elif self._corpus_type == 'extended':
            self._corpus = load_extended_raw_corpus()
            self.total = 36131
        elif self._corpus_type == 'tokenization':
            raise NotImplementedError('Tokenization Corpus is not yet implemented.')

    def __iter__(self):
        for document in tqdm(self._corpus, total=self.total):
            with tokenizer_context(self.toker):
                d = Doc(document)
                for sentence in d:
                    tokens = []
                    for token in sentence.tokens:
                        if token not in puncts:
                            tokens.append(tr_lower(token))
                    yield tokens
