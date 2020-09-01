from abc import ABC, abstractmethod
from typing import List
from .word_tokenizer_helper import word_tokenize
import warnings


class WordTokenizer(ABC):
    __instances = {}

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        pass

    def __call__(self, sentence: str) -> List[str]:
        return self._tokenize(str(sentence))

    @staticmethod
    def factory(tokenizer_name: str):
        normalized_name = tokenizer_name.lower().replace(' ', '').replace('-', '').replace('tokenizer', '')
        if normalized_name not in WordTokenizer.__instances:
            if normalized_name == "bert":
                WordTokenizer.__instances[normalized_name] = BertTokenizer()
            elif normalized_name == "simple":
                warnings.warn(
                    ("Note that SimpleTokenizer is pretty new in sadedeGel. "
                     "If you experience any problems, open up a issue "
                     "(https://github.com/GlobalMaksimum/sadedegel/issues/new)"))
                WordTokenizer.__instances[normalized_name] = SimpleTokenizer()
            else:
                raise Exception(
                    (f"No word tokenizer type match with name {tokenizer_name}."
                     " Use one of 'bert-tokenizer', 'SimpleTokenizer', etc."))

        return WordTokenizer.__instances[normalized_name]


class BertTokenizer(WordTokenizer):
    __name__ = "BertTokenizer"

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def __init__(self):
        super(BertTokenizer, self).__init__()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


class SimpleTokenizer(WordTokenizer):
    __name__ = "SimpleTokenizer"

    def __init__(self):
        super(SimpleTokenizer, self).__init__()
        self.tokenizer = word_tokenize

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)

    def convert_tokens_to_ids(self, ids: List[str]) -> List[int]:
        raise NotImplementedError("convert_tokens_to_ids is not implemented for SimpleTokenizer yet. Use BERTTokenizer")


def get_default_word_tokenizer() -> WordTokenizer:
    return WordTokenizer.factory(BertTokenizer.__name__)
