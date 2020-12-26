from collections import Counter
import re
from typing import List, Union
import warnings
from functools import partial

import torch

import numpy as np  # type:ignore

from loguru import logger
from scipy.sparse import csr_matrix

from ..ml.sbd import load_model
from ..metrics import rouge1_score
from .util import tr_lower, select_layer, __tr_lower_abbrv__, flatten, pad
from ..config import load_config
from .word_tokenizer import WordTokenizer
from .token import Token, IDF_METHOD_VALUES
from ..about import __version__


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


TF_BINARY, TF_RAW, TF_FREQ, TF_LOG_NORM, TF_DOUBLE_NORM = "binary", "raw", "freq", "log_norm", "double_norm"
TF_METHOD_VALUES = [TF_BINARY, TF_RAW, TF_FREQ, TF_LOG_NORM, TF_DOUBLE_NORM]


class Sentences:

    def __init__(self, id_: int, text: str, doc, config: dict = {}):
        self.id = id_
        self.text = text

        self._tokens = None
        self.document = doc
        self._bert = None
        self.toks = None

        # No failback to config read in here because this will slow down sentence instantiation extremely.
        tf_method = config['tf']['method']

        if tf_method == TF_BINARY:
            self.f_tf = self.binary_tf
        elif tf_method == TF_RAW:
            self.f_tf = self.raw_tf
        elif tf_method == TF_FREQ:
            self.f_tf = self.freq_tf
        elif tf_method == TF_LOG_NORM:
            self.f_tf = self.log_norm_tf
        elif tf_method == TF_DOUBLE_NORM:
            k = config.getfloat('tf', 'double_norm_k')

            if not 0 < k < 1:
                raise ValueError(
                    f"Invalid k value {k} for double norm term frequency. Values should be between 0 and 1.")
            self.f_tf = partial(self.double_norm_tf, k=k)
        else:
            raise ValueError(f"Unknown term frequency method {tf_method}. Choose on of {','.join(TF_METHOD_VALUES)}")

    @property
    def tokenizer(self):
        return self.document.tokenizer

    @property
    def vocabulary(self):
        return self.tokenizer.vocabulary

    @classmethod
    def set_tf_function(cls, tf_type):
        raise DeprecationWarning("Function is depreciated.")

    @property
    def bert(self):
        return self._bert

    @bert.setter
    def bert(self, bert):
        self._bert = bert

    @property
    def input_ids(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokens_with_special_symbols)

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = self.tokenizer(self.text)

        return self._tokens

    @property
    def tokens_with_special_symbols(self):
        return ['[CLS]'] + self.tokens + ['[SEP]']

    def rouge1(self, metric):
        return rouge1_score(
            flatten([[tr_lower(token) for token in sent.tokens] for sent in self.document if sent.id != self.id]),
            [tr_lower(t) for t in self.tokens], metric)

    @property
    def _doc_toks(self):
        return dict(Counter([tok for sent in self.document for tok in sent.tokens]))

    @property
    def _doc_len(self):
        return len([tok for sent in self.document for tok in sent.tokens])

    def tfidf(self):
        return self.tf * self.idf

    @property
    def tf(self):
        return self.f_tf()

    def binary_tf(self):
        return self.raw_tf().clip(max=1)

    def raw_tf(self):
        v = np.zeros(len(self.vocabulary))

        for token in self.tokens:
            t = self.vocabulary[token]
            if not t.is_oov:
                v[t.id] = self._doc_toks[token]

        return v

    def freq_tf(self):
        return self.raw_tf() / self.document.raw_tf().sum()

    def log_norm_tf(self):
        return np.log1p(self.raw_tf())

    def double_norm_tf(self, k=0.5):
        if not (0 < k < 1):
            raise ValueError(f"Ensure that 0 < k < 1 for double normalization term frequency calculation")

        return k + (1 - k) * (self.raw_tf() / self.document.raw_tf().max())

    @property
    def idf(self):
        v = np.zeros(len(self.vocabulary))

        for token in self.tokens:
            t = self.vocabulary[token]
            if not t.is_oov:
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

    def __getitem__(self, token_ix):
        return self.vocabulary[self.tokens[token_ix]]

    def __iter__(self):
        for t in self.tokens:
            yield self.vocabulary[t]


class Document:
    def __init__(self, raw, builder):
        self.raw = raw
        self.spans = []
        self._sents = []
        self._bert = None
        self.builder = builder

    @property
    def vocabulary(self):
        return self.tokenizer.vocabulary

    @property
    def tokenizer(self):
        return self.builder.tokenizer

    @property
    def sents(self):
        if tuple(map(int, __version__.split('.'))) < (0, 17):
            warnings.warn(
                ("Doc.sents is deprecated and will be removed by 0.17. "
                 "Use either iter(Doc) or Doc[i] to access specific sentences in document."), DeprecationWarning,
                stacklevel=2)
        else:
            raise Exception("Remove .sent before release.")

        return self._sents

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

            if DocBuilder.bert_model is None:
                logger.info("Loading BertModel")
                from transformers import BertModel

                DocBuilder.bert_model = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased",
                                                                  output_hidden_states=True)
                DocBuilder.bert_model.eval()

            with torch.no_grad():
                outputs = DocBuilder.bert_model(inp, mask)

            twelve_layers = outputs[2][1:]

            self._bert = select_layer(twelve_layers, [11], return_cls=False)

        return self._bert

    @property
    def tfidf_embeddings(self):

        indptr = [0]
        indices = []
        data = []
        for i in range(len(self)):
            sent_embedding = self[i].tfidf()
            for idx in sent_embedding.nonzero()[0]:
                indices.append(idx)
                data.append(sent_embedding[idx])

            indptr.append(len(indices))

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), len(self.vocabulary)))

        return m

    @property
    def tf(self):
        indptr = [0]
        indices = []
        data = []
        for i in range(len(self)):
            _tf = self[i].tf
            for idx in _tf.nonzero()[0]:
                indices.append(idx)
                data.append(_tf[idx])

            indptr.append(len(indices))

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), len(self.vocabulary)))

        return m.max(axis=0)

    def raw_tf(self):
        v = np.zeros(len(self.vocabulary))

        for s in self:
            v += s.raw_tf()

        return v

    @property
    def idf(self):
        indptr = [0]
        indices = []
        data = []
        for i in range(len(self.sents)):
            idf = self.sents[i].idf
            for idx in idf.nonzero()[0]:
                indices.append(idx)
                data.append(idf[idx])

            indptr.append(len(indices))

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), len(self.vocabulary)))

        return m.max(axis=0)

    def tfidf(self):
        return self.tf.multiply(self.idf)

    def from_sentences(self, sentences: List[str]):
        return self.builder.from_sentences(sentences)


class DocBuilder:
    bert_model = None

    def __init__(self, **kwargs):

        self.config = load_config(kwargs)

        self.sbd = load_model()

        self.tokenizer = WordTokenizer.factory(self.config['default']['tokenizer'])

        idf_method = self.config['idf']['method']

        if idf_method in IDF_METHOD_VALUES:
            Token.config = self.config
        else:
            raise ValueError(f"Unknown term frequency method {idf_method}. Choose on of {','.join(idf_method)}")

    def __call__(self, raw):

        if raw is not None:
            _spans = [match.span() for match in re.finditer(r"\S+", raw)]

            d = Document(raw, self)
            d.spans = [Span(i, span, d) for i, span in enumerate(_spans)]

            if len(d.spans) > 0:
                y_pred = self.sbd.predict((span.span_features() for span in d.spans))
            else:
                y_pred = []

            eos_list = [end for (start, end), y in zip(_spans, y_pred) if y == 1]

            if len(eos_list) > 0:
                for i, eos in enumerate(eos_list):
                    if i == 0:
                        d._sents.append(Sentences(i, d.raw[:eos].strip(), d, self.config))
                    else:
                        d._sents.append(Sentences(i, d.raw[eos_list[i - 1] + 1:eos].strip(), d, self.config))
            else:
                d._sents.append(Sentences(0, d.raw.strip(), d, self.config))
        else:
            raise Exception(f"{raw} document text can't be None")

        return d

    def from_sentences(self, sentences: List[str]):
        raw = "\n".join(sentences)

        d = Document(raw, self)
        for i, s in enumerate(sentences):
            d._sents.append(Sentences(i, s, d, self.config))

        return d
