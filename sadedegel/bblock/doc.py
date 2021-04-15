import re
import sys
from collections import Counter
from functools import partial
from typing import List

import numpy as np  # type:ignore
from rich.console import Console
from scipy.sparse import csr_matrix
from cached_property import cached_property

from .token import Token, IDF_METHOD_VALUES, IDFImpl
from .util import tr_lower, select_layer, __tr_lower_abbrv__, flatten, pad, normalize_tokenizer_name
from .word_tokenizer import WordTokenizer
from ..config import load_config
from ..metrics import rouge1_score
from ..ml.sbd import load_model

console = Console()


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


class TFImpl:
    def __init__(self):
        pass

    def raw_tf(self, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False) -> np.ndarray:
        if lowercase:
            v = np.zeros(self.vocabulary.size)
        else:
            v = np.zeros(self.vocabulary.size_cs)

        if lowercase:
            tokens = [t.lower_ for t in self.tokens]
        else:
            tokens = [t.word for t in self.tokens]

        counter = Counter(tokens)

        for token in tokens:
            t = Token(token)
            if t.is_oov or (drop_stopwords and t.is_stopword) or (drop_suffix and t.is_suffix) or (
                    drop_punct and t.is_punct):
                continue

            if lowercase:
                v[t.id] = counter[token]
            else:
                v[t.id_cs] = counter[token]

        return v

    def binary_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        return self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct).clip(max=1)

    def freq_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        tf = self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct)

        normalization = tf.sum()

        if normalization > 0:
            return tf / normalization
        else:
            return tf

    def log_norm_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False) -> np.ndarray:
        return np.log1p(self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct))

    def double_norm_tf(self, drop_stopwords=False, lowercase=False, drop_prefix=False, drop_punct=False,
                       k=0.5) -> np.ndarray:
        if not (0 < k < 1):
            raise ValueError(f"Ensure that 0 < k < 1 for double normalization term frequency calculation ({k} given)")

        tf = self.raw_tf(drop_stopwords, lowercase, drop_prefix, drop_punct)
        normalization = tf.max()

        if normalization > 0:
            return k + (1 - k) * (tf / normalization)
        else:
            return tf

    def get_tf(self, method=TF_BINARY, drop_stopwords=False, lowercase=False, drop_suffix=False, drop_punct=False,
               **kwargs) -> np.ndarray:
        if method == TF_BINARY:
            return self.binary_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_RAW:
            return self.raw_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_FREQ:
            return self.freq_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_LOG_NORM:
            return self.log_norm_tf(drop_stopwords, lowercase, drop_suffix, drop_punct)
        elif method == TF_DOUBLE_NORM:
            return self.double_norm_tf(drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)
        else:
            raise ValueError(f"Unknown tf method ({method}). Choose one of {TF_METHOD_VALUES}")

    @property
    def tf(self):
        tf = self.config['tf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        return self.get_tf(tf, drop_stopwords, lowercase, drop_suffix, drop_punct)


class BM25Impl:
    def __init__(self):
        pass

    def get_bm25(self, tf_method: str, idf_method: str, k1: float, b: float, delta: float = 0, drop_stopwords=False,
                 lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        """Return bm25 embedding

        Refer to https://en.wikipedia.org/wiki/Okapi_BM25 for details

        :param tf_method:
        :param idf_method:
        :param k1:
        :param b:
        :param delta:
        :param drop_stopwords:
        :param lowercase:
        :param drop_suffix:
        :param drop_punct:
        :param kwargs:
        :return:
        """

        tf = self.get_tf(tf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)
        idf = self.get_idf(idf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)

        bm25 = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(self) / self.avgdl))) + delta)

        return bm25

    def get_bm11(self, tf_method, idf_method, k1, avgdl, delta=0, drop_stopwords=False, lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        return self.get_bm25(tf_method, idf_method, k1, 1, avgdl, delta, drop_stopwords, lowercase, drop_suffix,
                             drop_punct,
                             kwargs)

    def get_bm15(self, tf_method, idf_method, k1, avgdl, delta=0, drop_stopwords=False, lowercase=False,
                 drop_suffix=False,
                 drop_punct=False, **kwargs):
        return self.get_bm25(tf_method, idf_method, k1, 0, avgdl, delta, drop_stopwords, lowercase, drop_suffix,
                             drop_punct,
                             kwargs)


class Sentences(TFImpl, IDFImpl, BM25Impl):

    def __init__(self, id_: int, text: str, doc, config: dict = {}):
        TFImpl.__init__(self)
        IDFImpl.__init__(self)
        BM25Impl.__init__(self)

        self.id = id_
        self.text = text

        self.document = doc
        self.config = doc.builder.config
        self._bert = None

        # No fail back to config read in here because this will slow down sentence instantiation extremely.
        self.tf_method = config['tf']['method']

        if self.tf_method == TF_BINARY:
            self.f_tf = self.binary_tf
        elif self.tf_method == TF_RAW:
            self.f_tf = self.raw_tf
        elif self.tf_method == TF_FREQ:
            self.f_tf = self.freq_tf
        elif self.tf_method == TF_LOG_NORM:
            self.f_tf = self.log_norm_tf
        elif self.tf_method == TF_DOUBLE_NORM:
            k = config.getfloat('tf', 'double_norm_k')

            if not 0 < k < 1:
                raise ValueError(
                    f"Invalid k value {k} for double norm term frequency. Values should be between 0 and 1.")
            self.f_tf = partial(self.double_norm_tf, k=k)
        else:
            raise ValueError(
                f"Unknown term frequency method {self.tf_method}. Choose on of {','.join(TF_METHOD_VALUES)}")

    @property
    def avgdl(self) -> float:
        """Average number of tokens per sentence"""
        return self.config['default'].getfloat('avg_sentence_length')

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

    @cached_property
    def tokens(self) -> List[Token]:
        return [t for t in self.tokenizer(self.text)]

    @property
    def tokens_with_special_symbols(self):
        return [Token('[CLS]')] + self.tokens + [Token('[SEP]')]

    def rouge1(self, metric) -> float:
        return rouge1_score(
            flatten([[t.lower_ for t in sent] for sent in self.document if sent.id != self.id]),
            [t.lower_ for t in self], metric)

    @property
    def bm25(self) -> np.float32:

        tf = self.config['tf']['method']
        idf = self.config['idf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        k1 = self.config['bm25'].getfloat('k1')
        b = self.config['bm25'].getfloat('b')

        delta = self.config['bm25'].getfloat('delta')

        return np.sum(self.get_bm25(tf, idf, k1, b, delta, drop_stopwords, lowercase, drop_suffix, drop_punct),
                      dtype=np.float32)

    @property
    def tfidf(self):
        tf = self.config['tf']['method']
        idf = self.config['idf']['method']
        drop_stopwords = self.config['default'].getboolean('drop_stopwords')
        lowercase = self.config['default'].getboolean('lowercase')
        drop_suffix = self.config['bert'].getboolean('drop_suffix')
        drop_punct = self.config['default'].getboolean('drop_punct')

        return self.get_tf(tf, drop_stopwords, lowercase, drop_suffix, drop_punct) * self.get_idf(idf, drop_stopwords,
                                                                                                  lowercase,
                                                                                                  drop_suffix,
                                                                                                  drop_punct)

    def get_tfidf(self, tf_method, idf_method, drop_stopwords=False, lowercase=False, drop_suffix=False,
                  drop_punct=False, **kwargs) -> np.ndarray:
        return self.get_tf(tf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs) * self.get_idf(
            idf_method, drop_stopwords, lowercase, drop_suffix, drop_punct, **kwargs)

    @property
    def tf(self):
        return self.f_tf()

    @property
    def idf(self):
        v = np.zeros(len(self.vocabulary))

        for t in self.tokens:
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
        return self.tokens[token_ix]

    def __iter__(self):
        yield from self.tokens


class Document(TFImpl, IDFImpl, BM25Impl):
    def __init__(self, raw, builder):
        TFImpl.__init__(self)
        IDFImpl.__init__(self)
        BM25Impl.__init__(self)

        self.raw = raw
        self.spans = []
        self._sents = []
        self._tokens = None
        self.builder = builder
        self.config = self.builder.config

    @property
    def avgdl(self) -> float:
        """Average number of tokens per document"""
        return self.config['default'].getfloat('avg_document_length')

    @cached_property
    def tokens(self) -> List[str]:
        tokens = []
        for s in self:
            for t in s.tokens:
                tokens.append(t)

        return tokens

    @property
    def vocabulary(self):
        return self.tokenizer.vocabulary

    @property
    def tokenizer(self):
        return self.builder.tokenizer

    def __getitem__(self, key):
        return self._sents[key]

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
            try:
                import torch
            except ImportError:
                console.print(
                    ("Error in importing transformers module. "
                     "Ensure that you run 'pip install sadedegel[bert]' to use BERT features."))
                sys.exit(1)

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

    @cached_property
    def bert_embeddings(self):
        try:
            import torch
            from transformers import BertModel
        except ImportError:
            console.print(
                ("Error in importing transformers module. "
                 "Ensure that you run 'pip install sadedegel[bert]' to use BERT features."))
            sys.exit(1)

        inp, mask = self.padded_matrix()

        if DocBuilder.bert_model is None:
            DocBuilder.bert_model = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased",
                                                              output_hidden_states=True)
            DocBuilder.bert_model.eval()

        with torch.no_grad():
            outputs = DocBuilder.bert_model(inp, mask)

        twelve_layers = outputs[2][1:]

        return select_layer(twelve_layers, [11], return_cls=False)

    def get_tfidf(self, tf_method, idf_method, **kwargs):
        return self.get_tf(tf_method, **kwargs) * self.get_idf(idf_method, **kwargs)

    @property
    def tfidf(self):
        return self.tf * self.idf

    @property
    def tfidf_matrix(self):
        indptr = [0]
        indices = []
        data = []

        for i in range(len(self)):
            sent_embedding = self[i].tfidf

            for idx in sent_embedding.nonzero()[0]:
                indices.append(idx)
                data.append(sent_embedding[idx])

            indptr.append(len(indices))

        lowercase = self.config['default'].getboolean('lowercase')

        if lowercase:
            dim = self.vocabulary.size
        else:
            dim = self.vocabulary.size_cs

        m = csr_matrix((data, indices, indptr), dtype=np.float32, shape=(len(self), dim))

        return m

    def from_sentences(self, sentences: List[str]):
        return self.builder.from_sentences(sentences)


class DocBuilder:
    bert_model = None

    def __init__(self, **kwargs):

        self.config = load_config(kwargs)

        self.sbd = load_model()

        tokenizer_str = normalize_tokenizer_name(self.config['default']['tokenizer'])

        self.tokenizer = WordTokenizer.factory(tokenizer_str, emoji=self.config['tokenizer'].getboolean('emoji'),
                                               hashtag=self.config['tokenizer'].getboolean('hashtag'),
                                               mention=self.config['tokenizer'].getboolean('mention'))

        Token.set_vocabulary(self.tokenizer.vocabulary)

        self.config['default']['avg_sentence_length'] = self.config[tokenizer_str]['avg_sentence_length']
        self.config['default']['avg_document_length'] = self.config[tokenizer_str]['avg_document_length']

        idf_method = self.config['idf']['method']

        if idf_method in IDF_METHOD_VALUES:
            Token.config = self.config
        else:
            raise ValueError(f"Unknown term frequency method {idf_method}. Choose on of {IDF_METHOD_VALUES}")

    def __call__(self, raw):

        if raw is not None:
            raw_stripped = raw.strip()
            _spans = [match.span() for match in re.finditer(r"\S+", raw_stripped)]

            d = Document(raw_stripped, self)
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

                if eos_list[-1] != len(raw_stripped):
                    d._sents.append(Sentences(len(d._sents), d.raw[eos_list[-1] + 1:len(raw_stripped)], d,
                                              self.config))
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
