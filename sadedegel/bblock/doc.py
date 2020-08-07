from transformers import AutoTokenizer

import re
from typing import List
import numpy as np
import torch
from sadedegel.metrics import rouge1_score
from loguru import logger

from ..ml.sbd import load_model
from .util import tr_lower, select_layer, __tr_lower_abbrv__, flatten, pad


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

        """NOTE: 'Beşiktaş’ın' is not title by python :/
        """
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
                features[f"PREFIX_IS_ABBRV"] = True
            if any(tr_lower(word).endswith(abbrv) for abbrv in __tr_lower_abbrv__):
                features[f"SUFFIX_IS_ABBRV"] = True
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
    tokenizer = None

    def __init__(self, id_: int, text: str, all_sentences: List):
        if Sentences.tokenizer is None:
            Sentences.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        self.id = id_
        self.text = text

        self._input_ids = None
        self._tokens = None
        self.all_sentences = all_sentences
        self._bert = None

    @property
    def bert(self):
        return self._bert

    @bert.setter
    def bert(self, bert):
        self._bert = bert

    @property
    def input_ids(self):
        if self._input_ids is None:
            self._input_ids = Sentences.tokenizer(self.text)['input_ids']

        return self._input_ids

    @property
    def tokens(self):
        return self.tokens_with_special_symbols[1:-1]

    @property
    def tokens_with_special_symbols(self):
        self._tokens = Sentences.tokenizer.convert_ids_to_tokens(self.input_ids)

        return self._tokens

    def rouge1(self, metric):
        return rouge1_score(
            flatten([[tr_lower(token) for token in sent.tokens] for sent in self.all_sentences if sent.id != self.id]),
            [tr_lower(t) for t in self.tokens], metric)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)


class Doc:
    sbd = None
    model = None

    def __init__(self, raw: str, sents=None):
        if Doc.sbd is None and sents is None:
            logger.info("Loading ML based SBD")
            Doc.sbd = load_model()

        self.raw = raw
        self._bert = None
        self.sents = []

        if sents is None:
            _spans = [match.span() for match in re.finditer(r"\S+", self.raw)]

            self.spans = [Span(i, span, self) for i, span in enumerate(_spans)]

            y_pred = Doc.sbd.predict((span.span_features() for span in self.spans))

            eos_list = [end for (start, end), y in zip(_spans, y_pred) if y == 1]

            if len(eos_list) > 0:
                for i, eos in enumerate(eos_list):
                    if i == 0:
                        self.sents.append(Sentences(i, self.raw[:eos].strip(), self.sents))
                    else:
                        self.sents.append(Sentences(i, self.raw[eos_list[i - 1] + 1:eos].strip(), self.sents))
            else:
                self.sents.append(Sentences(0, self.raw.strip(), self.sents))

        else:
            for i, s in enumerate(sents):
                self.sents.append(Sentences(i, s, self.sents))

    def __str__(self):
        return self.raw

    def __repr__(self):
        return self.raw

    def __len__(self):
        return len(self.sents)

    def max_length(self):
        """Maximum length of a sentence including special symbols."""
        return max(len(s.tokens_with_special_symbols) for s in self.sents)

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
            mat = torch.tensor([pad(s.input_ids, max_len) for s in self.sents])

            if return_mask:
                return mat, (mat > 0).to(int)
            else:
                return mat
        else:
            mat = np.array([pad(s.input_ids, max_len) for s in self.sents])

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
