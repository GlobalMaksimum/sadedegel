import sys
import re
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

from cached_property import cached_property
from rich.console import Console

from sadedegel.bblock.word_tokenizer_helper import ICUTokenizerHelper
from .util import normalize_tokenizer_name
from .vocabulary import Vocabulary
from .token import Token
from .word_tokenizer_helper import word_tokenize
from ..about import __version__


class TokenType(Enum):
    TEXT = "text"
    MENTION = "mention"
    EMOJI = "emoji"
    HASHTAG = "hashtag"


@dataclass
class TokenSpan:
    type: TokenType
    start: int
    end: int


console = Console()


class WordTokenizer(ABC):
    __instances = {}

    def __init__(self, mention=False, hashtag=False, emoji=False):
        """

        @param mention: Handle mention in tweet texts.
        @param hashtag: Handle hashtag in tweet texts.
        @param emoji: Handle emoji unicode texts in texts.
        """
        self._vocabulary = None
        self.mention = mention
        self.hashtag = hashtag
        self.emoji = emoji

        self.regexes = []

        if self.hashtag:
            console.print("Handling hashtags")
            self.regexes.append(re.compile(r"(?P<hashtag>#\S+)"))

        if self.mention:
            console.print("Handling mentions")
            self.regexes.append(re.compile(r"(?P<mention>@\S+)"))

        if self.emoji:
            self.regexes.append(re.compile(r"(?P<emoji>[\U00010000-\U0010ffff])",
                                           flags=re.UNICODE))

        if len(self.regexes) > 0:
            self.exception_rules = re.compile('|'.join(x.pattern for x in self.regexes), flags=re.UNICODE)

        console.log(f"{len(self.regexes)} tokenizer exception rules.")

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        pass

    def __call__(self, sentence: str) -> List[Token]:
        text = str(sentence)

        if len(self.regexes) == 0:
            return [Token(t) for t in self._tokenize(text)]
        else:
            EOS = len(text)

            spans = []
            for m in self.exception_rules.finditer(text):
                start, end = m.start(), m.end()

                if len(spans) == 0:
                    if start != 0:
                        spans.append(TokenSpan(TokenType.TEXT, 0, start))
                else:
                    if start > spans[-1].end:
                        spans.append(TokenSpan(TokenType.TEXT, spans[-1].end, start))

                if m.lastgroup == "hashtag":
                    spans.append(TokenSpan(TokenType.HASHTAG, start, end))
                elif m.lastgroup == "mention":
                    spans.append(TokenSpan(TokenType.MENTION, start, end))
                else:
                    spans.append(TokenSpan(TokenType.EMOJI, start, end))

            if len(spans) == 0:
                if EOS != 0:
                    spans.append(TokenSpan(TokenType.TEXT, 0, EOS))
            else:
                if EOS > spans[-1].end:
                    spans.append(TokenSpan(TokenType.TEXT, spans[-1].end, EOS))

            tokens = []
            for s in spans:
                if s.type == TokenType.TEXT:
                    tokens += [Token(t) for t in self._tokenize(text[s.start:s.end])]
                elif s.type == TokenType.EMOJI:
                    t = Token(text[s.start:s.end])
                    t.is_emoji = True
                    tokens.append(t)
                elif s.type == TokenType.HASHTAG:
                    t = Token(text[s.start:s.end])
                    t.is_hashtag = True
                    tokens.append(t)
                else:
                    t = Token(text[s.start:s.end])
                    t.is_mention = True
                    tokens.append(t)

            return tokens

    @staticmethod
    def factory(tokenizer_name: str, mention=False, hashtag=False, emoji=False):
        console.log(f"mention={mention}, hashtag={hashtag}, emoji={emoji}")
        normalized_name = normalize_tokenizer_name(tokenizer_name)
        if normalized_name not in WordTokenizer.__instances:
            if normalized_name == "bert":
                return BertTokenizer(mention, hashtag, emoji)
            elif normalized_name == "simple":
                warnings.warn(
                    ("Note that SimpleTokenizer is pretty new in sadedeGel. "
                     "If you experience any problems, open up a issue "
                     "(https://github.com/GlobalMaksimum/sadedegel/issues/new)"))
                return SimpleTokenizer(mention, hashtag, emoji)
            elif normalized_name == "icu":
                return ICUTokenizer(mention, hashtag, emoji)
            else:
                raise Exception(
                    (f"No word tokenizer type match with name {tokenizer_name}."
                     " Use one of 'bert-tokenizer', 'SimpleTokenizer', etc."))

        # return WordTokenizer.__instances[normalized_name]


class BertTokenizer(WordTokenizer):
    __name__ = "BertTokenizer"

    def convert_tokens_to_ids(self, tokens: List[Token]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids([t.word for t in tokens])

    def __init__(self, mention=False, hashtag=False, emoji=False):
        super(BertTokenizer, self).__init__(mention, hashtag, emoji)

        self.tokenizer = None

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            try:
                import torch
                from transformers import AutoTokenizer
            except ImportError:
                console.print(
                    ("Error in importing transformers module. "
                     "Ensure that you run 'pip install sadedegel[bert]' to use BERT features."))
                sys.exit(1)
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        return self.tokenizer.tokenize(text)

    @cached_property
    def vocabulary(self):
        try:
            return Vocabulary("bert")
        except FileNotFoundError:
            console.print("[red]bert[/red] vocabulary file not found.")

            return None


class SimpleTokenizer(WordTokenizer):
    __name__ = "SimpleTokenizer"

    def __init__(self, mention=False, hashtag=False, emoji=False):
        super(SimpleTokenizer, self).__init__(mention, hashtag, emoji)
        self.tokenizer = word_tokenize

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)

    def convert_tokens_to_ids(self, ids: List[str]) -> List[int]:
        raise NotImplementedError("convert_tokens_to_ids is not implemented for SimpleTokenizer yet. Use BERTTokenizer")

    @cached_property
    def vocabulary(self):
        try:
            return Vocabulary("simple")
        except FileNotFoundError:
            console.print("[red]simple[/red] vocabulary file not found.")

            return None


class ICUTokenizer(WordTokenizer):
    __name__ = "ICUTokenizer"

    def __init__(self, mention=False, hashtag=False, emoji=False):
        super(ICUTokenizer, self).__init__(mention, hashtag, emoji)
        self.tokenizer = ICUTokenizerHelper()

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)

    def convert_tokens_to_ids(self, ids: List[str]) -> List[int]:
        raise NotImplementedError("convert_tokens_to_ids is not implemented for SimpleTokenizer yet. Use BERTTokenizer")

    @cached_property
    def vocabulary(self):
        try:
            return Vocabulary("icu")
        except FileNotFoundError:
            console.print("[red]icu[/red] vocabulary file not found.")

            return None
