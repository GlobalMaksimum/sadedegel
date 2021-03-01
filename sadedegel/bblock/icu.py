from icu import Locale, BreakIterator
from typing import List


class ICUTokenizerHelper:
    def __init__(self):
        self.locale = Locale("tr")
        self.breakor = BreakIterator.createWordInstance(self.locale)

    def __call__(self, text: str) -> List[str]:
        self.breakor.setText(text)

        parts = []
        p0 = 0
        for p1 in self.breakor:
            part = text[p0:p1].strip()
            if len(part) > 0:
                parts.append(part)
            p0 = p1

        return parts
