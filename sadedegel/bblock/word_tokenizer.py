from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict
from .word_tokenizer_helper import word_tokenize


class BaseTokenizer(ABC):

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        pass

    def tokenize(self, text: str) -> List[str]:
        result = self._tokenize(text)

        return self.to_tokens(result)

    def __call__(self, sentence: str) -> List[dict]:
        return self._tokenize(str(sentence))


class BertTokenizer(BaseTokenizer):
    name = "BERT Tokenizer"

    def __init__(self, keep_special_tokens=True):
        super(BertTokenizer, self).__init__()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def to_tokens(self, tokens: Dict[str, List[Union[int, str]]]) -> List[str]:
        return self.convert_ids_to_tokens(tokens['input_ids'])[1:-1]

    def _tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        return self.tokenizer(text)

    def __str__(self):
        return self.name


class SimpleTokenizer(BaseTokenizer):
    name = "Simple Tokenizer"

    def __init__(self):
        super(SimpleTokenizer, self).__init__()
        self.tokenizer = word_tokenize

    def _tokenize(self, text: str) -> List[str]:
        return dict(tokens=self.tokenizer(text))

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        raise NotImplementedError("convert_ids_to_tokens is not implemented for SimpleTokenizer yet.")

    def to_tokens(self, tokens: Dict[str, List[Union[int, str]]]) -> List[str]:
        return tokens['tokens']

    def __str__(self) -> str:
        return self.name


def get_tokenizer_instance_by_name(tokenizer_name: str) -> BaseTokenizer:
    if tokenizer_name == SimpleTokenizer.name:
        return SimpleTokenizer()
    elif tokenizer_name == BertTokenizer.name:
        return BertTokenizer()
    else:
        raise ValueError(f"No word tokenizer is available with name: {tokenizer_name}")


def get_default_word_tokenizer() -> BaseTokenizer:
    return get_tokenizer_instance_by_name(BertTokenizer.name)
