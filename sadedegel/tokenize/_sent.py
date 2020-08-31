from abc import ABCMeta, abstractmethod
from typing import List

import re
import nltk

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


class SentencesTokenizer(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, doc: str) -> List[str]:
        return self._split(doc)

    @abstractmethod
    def _split(self, text):
        pass


class RegexpSentenceTokenizer(SentencesTokenizer):

    def _split(self, text: str) -> List[str]:
        text = " " + text + "  "
        text = text.replace(r"\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)

        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")

        text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)

        if "”" in text:
            text = text.replace(".”", "”.")

        if "\"" in text:
            text = text.replace(".\"", "\".")

        if "!" in text:
            text = text.replace("!\"", "\"!")

        if "?" in text:
            text = text.replace("?\"", "\"?")

        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences if len(s) > 1]

        return sentences


class NLTKPunctTokenizer(SentencesTokenizer):
    def __init__(self):
        super().__init__()

        self.sent_detector = nltk.data.load('tokenizers/punkt/turkish.pickle')

    def _split(self, text: str) -> List[str]:
        return self.sent_detector.tokenize(text)
