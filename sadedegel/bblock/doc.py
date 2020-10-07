import re
from typing import List, Union
import warnings

import torch

import numpy as np  # type:ignore

from loguru import logger
from scipy.sparse import csr_matrix

from ..ml.sbd import load_model
from ..metrics import rouge1_score
from .util import tr_lower, select_layer, __tr_lower_abbrv__, flatten, pad
from .word_tokenizer import get_default_word_tokenizer, WordTokenizer
from .vocabulary import Token


class Span:
    def __init__(self, i: int, span, doc):
        self.doc = doc
        self.i = i
        self.value = span

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    @property
    def text(self):
        return self.doc.raw[slice(*self.value)]

    def span_features(self):
        is_first_span = self.i == 0
        is_last_span = self.i == len(self.doc.spans) - 1
        word = self.doc.raw[slice(*self.value)]

        if not is_first_span:
            word_m1 = self.doc.raw[slice(*self.doc.spans[self.i - 1].value)]
        else:
            word_m1 = '<D>'

        if not is_last_span:
            word_p1 = self.doc.raw[slice(*self.doc.spans[self.i + 1].value)]
        else:
            word_p1 = '</D>'

        features = {}

        # All upper features
        if word_m1.isupper() and not is_first_span:
            features["PREV_ALL_CAPS"] = True

        if word.isupper():
            features["ALL_CAPS"] = True

        if word_p1.isupper() and not is_last_span:
            features["NEXT_ALL_CAPS"] = True

        # All lower features
        if word_m1.islower() and not is_first_span:
            features["PREV_ALL_LOWER"] = True

        if word.islower():
            features["ALL_LOWER"] = True

        if word_p1.islower() and not is_last_span:
            features["NEXT_ALL_LOWER"] = True

        # Century
        if tr_lower(word_p1).startswith("yüzyıl") and not is_last_span:
            features["NEXT_WORD_CENTURY"] = True

        # Number
        if (tr_lower(word_p1).startswith("milyar") or tr_lower(word_p1).startswith("milyon") or tr_lower(
                word_p1).startswith("bin")) and not is_last_span:
            features["NEXT_WORD_NUMBER"] = True

        # Percentage
        if tr_lower(word_m1).startswith("yüzde") and not is_first_span:
            features["PREV_WORD_PERCENTAGE"] = True

        # In parenthesis feature
        if word[0] == '(' and word[-1] == ')':
            features["IN_PARENTHESIS"] = True

        # Suffix features
        m = re.search(r'\W+$', word)

        if m:
            subspan = m.span()
            suff2 = word[slice(*subspan)][:2]

            if all((c in '!\').?’”":;…') for c in suff2):
                features['NON-ALNUM SUFFIX2'] = suff2

        if word_p1[0] in '-*◊' and not is_last_span:
            features["NEXT_BULLETIN"] = True

        # title features
        # if word_m1.istitle() and not is_first_span:
        #     features["PREV_TITLE"] = 1

        # NOTE: 'Beşiktaş’ın' is not title by python :/
        m = re.search(r'\w+', word)

        if m:
            subspan = m.span()
            if word[slice(*subspan)].istitle():
                features["TITLE"] = True

        m = re.search(r'\w+', word_p1)

        if m:
            subspan = m.span()

            if word_p1[slice(*subspan)].istitle() and not is_last_span:
                features["NEXT_TITLE"] = True

        # suffix features
        for name, symbol in zip(['DOT', 'ELLIPSES', 'ELLIPSES', 'EXCLAMATION', 'QUESTION'],
                                ['.', '...', '…', '?', '!']):
            if word.endswith(symbol):
                features[f"ENDS_WITH_{name}"] = True

        # prefix abbreviation features
        if '.' in word:

            if any(tr_lower(word).startswith(abbrv) for abbrv in __tr_lower_abbrv__):
                features["PREFIX_IS_ABBRV"] = True
            if any(tr_lower(word).endswith(abbrv) for abbrv in __tr_lower_abbrv__):
                features["SUFFIX_IS_ABBRV"] = True
            else:
                prefix = word.split('.', maxsplit=1)[0]
                features["PREFIX_IS_DIGIT"] = prefix.isdigit()

        # if '.' in word_m1:
        #     prefix_m1 = word_m1.split('.', maxsplit=1)[0]
        #
        #     if tr_lower(prefix_m1) in __tr_lower_abbrv__ and not is_first_span:
        #         features[f"PREV_PREFIX_IS_ABBRV"] = True
        #     else:
        #         features["PREV_PREFIX_IS_DIGIT"] = prefix_m1.isdigit()
        #
        # if '.' in word_p1:
        #     prefix_p1 = word_p1.split('.', maxsplit=1)[0]
        #
        #     if tr_lower(prefix_p1) in __tr_lower_abbrv__ and not is_last_span:
        #         features[f"NEXT_PREFIX_IS_ABBRV"] = True
        #     else:
        #         features["NEXT_PREFIX_IS_DIGIT"] = prefix_p1.isdigit()

        return features


class Sentences:
    tokenizer = get_default_word_tokenizer()
    vocabulary = Token.set_vocabulary(tokenizer)

    def __init__(self, id_: int, text: str, doc):
        self.id = id_
        self.text = text

        self._tokens = None
        self.document = doc
        self._bert = None
        self.toks = None

    @staticmethod
    def set_word_tokenizer(tokenizer_name):
        if tokenizer_name != Sentences.tokenizer.__name__:
            Sentences.tokenizer = WordTokenizer.factory(tokenizer_name)
            Sentences.vocabulary = Token.set_vocabulary(Sentences.tokenizer)

    @property
    def bert(self):
        return self._bert

    @bert.setter
    def bert(self, bert):
        self._bert = bert

    @property
    def input_ids(self):
        return Sentences.tokenizer.convert_tokens_to_ids(self.tokens_with_special_symbols)

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = Sentences.tokenizer(self.text)

        return self._tokens

    @property
    def tokens_with_special_symbols(self):
        return ['[CLS]'] + self.tokens + ['[SEP]']

    def rouge1(self, metric):
        return rouge1_score(
            flatten([[tr_lower(token) for token in sent.tokens] for sent in self.document if sent.id != self.id]),
            [tr_lower(t) for t in self.tokens], metric)

    def tfidf(self):
        return self.tf * self.idf

    @property
    def tf(self):
        v = np.zeros(Sentences.vocabulary.size)

        for token in self.tokens:
            t = Token(token)
            if t:
                v[t.id] = 1

        return v

    @property
    def idf(self):
        v = np.zeros(Sentences.vocabulary.size)

        for token in self.tokens:
            t = Token(token)
            if t:
                v[t.id] = t.idf

        return v

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    def __eq__(self, s: str):
        return self.text == s  # no need for type checking, will return false for non-strings


class Doc:
    sbd = None
    model = None

    def __init__(self, raw: Union[str, None]):
        if Doc.sbd is None and raw is not None:
            logger.info("Loading ML based SBD")
            Doc.sbd = load_model()

        self.raw = raw
        self._bert = None
        self._sents = []
        self.spans = None

        if raw is not None:
            _spans = [match.span() for match in re.finditer(r"\S+", self.raw)]

            self.spans = [Span(i, span, self) for i, span in enumerate(_spans)]

            y_pred = Doc.sbd.predict((span.span_features() for span in self.spans))

            eos_list = [end for (start, end), y in zip(_spans, y_pred) if y == 1]

            if len(eos_list) > 0:
                for i, eos in enumerate(eos_list):
                    if i == 0:
                        self._sents.append(Sentences(i, self.raw[:eos].strip(), self))
                    else:
                        self._sents.append(Sentences(i, self.raw[eos_list[i - 1] + 1:eos].strip(), self))
            else:
                self._sents.append(Sentences(0, self.raw.strip(), self))

    @property
    def sents(self):
        warnings.warn(
            ("Doc.sents is deprecated and will be removed by 0.17."
             "Use either iter(Doc) or Doc[i] to access specific sentences in document."), DeprecationWarning,
            stacklevel=2)

        return self._sents

    @classmethod
    def from_sentences(cls, sentences: List[str]):

        d = Doc(None)

        for i, s in enumerate(sentences):
            d._sents.append(Sentences(i, s, d))

        d.raw = "\n".join(sentences)

        return d

    def __getitem__(self, sent_idx):
        return self._sents[sent_idx]

    def __iter__(self):
        return iter(self._sents)

    def __str__(self):
        return self.raw

    def __repr__(self):
        return self.raw

    def __len__(self):
        return len(self._sents)

    def max_length(self):
        """Maximum length of a sentence including special symbols."""
        return max(len(s.tokens_with_special_symbols) for s in self._sents)

    def padded_matrix(self, return_numpy=False, return_mask=True):
        """Returns a 0 padded numpy.array or torch.tensor
              One row for each sentence
              One column for each token (pad 0 if length of sentence is shorter than the max length)

        :param return_numpy: Whether to return numpy.array or torch.tensor
        :param return_mask: Whether to return padding mask
        :return:
        """
        max_len = self.max_length()

        if not return_numpy:
            mat = torch.tensor([pad(s.input_ids, max_len) for s in self])

            if return_mask:
                return mat, (mat > 0).to(int)
            else:
                return mat
        else:
            mat = np.array([pad(s.input_ids, max_len) for s in self])

            if return_mask:
                return mat, (mat > 0).astype(int)
            else:
                return mat

    @property
    def bert_embeddings(self):
        if self._bert is None:
            inp, mask = self.padded_matrix()

            if Doc.model is None:
                logger.info("Loading BertModel")
                from transformers import BertModel

                Doc.model = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased", output_hidden_states=True)
                Doc.model.eval()

            with torch.no_grad():
                outputs = Doc.model(inp, mask)

            twelve_layers = outputs[2][1:]

            self._bert = select_layer(twelve_layers, [11], return_cls=False)

        return self._bert

    @property
    def tfidf_embeddings(self):

        indptr = [0]
        indices = []
        data = []
        for i in range(len(self.sents)):
            sent_embedding = self.sents[i].tfidf()
            for idx in sent_embedding.nonzero()[0]:
                indices.append(idx)
                data.append(sent_embedding[idx])

            indptr.append(len(indices))

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), Sentences.vocabulary.size))

        return m
