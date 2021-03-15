import re
import string
from typing import List

from icu import Locale, BreakIterator

from .util import space_pad

metatoken_telno = "phone"
metatoken_cctelno = "shortphone"
metatoken_hhmm = "hhmm"
metatoken_plate = "plate"
metatoken_ddmmyyyy = "ddmmyyyy"

mail_re = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*")

website_re = re.compile(
    r'\b(http[s]?\:\/\/)?[a-zı0-9]{3,}([./][a-zı0-9]*)+[./a-zı0-9]+(:[a-zı0-9]+)?')
phone_re = re.compile(
    r'\b(\(?0[ -]?[0-9]{3}\)?[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2}|0 ?[(]{1}[0-9]{3}[)][ -][0-9]{3}[ -][0-9]{2}[ -][0-9]{2}|[2|5][0-9]{2}[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2})|((0\s*)?21[26]\s+\d{3}\s+\d{2}\s+\d{2})')
cc_phone_re = re.compile(r'([444]{3}[ -]?[0-9]{1,2}[ -]?[0-9]{2,3})|(444\s*\d{2}\s*\d{2})')
# time_re = re.compile(
#    r'([0-9]{1,2}[:., ][0-9]{2}[ ]{0,2}[pm|am]|[0-9]{1,2}[:., ][0-9]{2})|((0[0-9]|1[0-9]|2[0-3])(:|\s+|\.)[0-5][0-9](:[0-5][0-9])?)')

time_re = re.compile(
    r"\b(?P<hour>1[0-9]|2[0-3]|0?[0-9])[:\.,](?P<minute>[1-5][0-9]|0?[0-9])(?:[:\.,](?P<second>[1-5][0-9]|0?[0-9]))?(?:[:\.,\s+](?P<pm_am>[ap]\.?m\.?))?")
carplate_re = re.compile(r"(?P<city>(0[1-9])|([1-7][0-9])|(8[01]))\s*(?P<mid>[A-Z][A-Z0-9]{1,4})\s*(?P<suffix>\d{2,4})")
# date_re = re.compile(
#    r'([0-9]{1,2}[./\-: ][0-9]{1,2}[./\-: ][0-9]{2,4}|[0-9]{1,2}[ ][A-ZİŞĞÖÜa-zışğöü]{1,10}[ ][0-9]{2,4}|[A-ZİŞĞÖÜa-zışğöü]{1,10}[ ][0-9]{1,2}[,][ ][0-9]{2,4})|(20\d{2}-\d{2}-\d{2})|(\d{2}[\.-:,]\d{2}[\.-:,]20\d{2})')
date_1_re = re.compile(
    r'\b(?P<day>0?[0-9]|1[0-9]|2[0-9]|3[01])(?:\s+|[:\-\./])(?P<month>oca(?:k)?|[sş]u(?:bat)?|mar(?:t)?|n[ıi]s(?:an)?|may(?:[ıi]s)?|haz(?:[iı]ran)?|tem(?:muz)?|a[gğ]u(?:stos)?|eyl(?:[uü]l)?|ek[iı](?:m)?|kas(?:[iı]m)?|ara(?:l[iı]k)?)(?:\s+|[:\-\./])(?P<year>\d{4})?',
    flags=re.IGNORECASE)
date_2_re = re.compile(
    r'\b(?P<day>0?[0-9]|1[0-9]|2[0-9]|3[01])(?:\s+|[:\-\./])(?P<month>(?:1[0-2]|0?[1-9]))(?:\s+|[:\-\./])(?P<year>\d{4})',
    flags=re.IGNORECASE)
space = re.compile(r"\s+")
token_re = re.compile(r"[@]?[a-zA-ZşŞiİıIğĞüÜöÖçÇ0-9]+|[\.]")

puncts = string.punctuation + '”“’‘…'


class Compose:
    def __init__(self, *funcs):
        self.debug = False
        self.funcs = funcs[0]

    def __call__(self, arg):
        if self.debug:
            print(arg)
        res = self.funcs[0](arg)

        if self.debug:
            print(res)

        for i in range(1, len(self.funcs)):
            res = self.funcs[i](res)
            if self.debug:
                print(self.funcs[i], res)

        return res


def compose(*funcs):
    return Compose(funcs)


c = compose(
    lambda t: mail_re.sub("email", t),
    lambda t: website_re.sub("website", t),
    lambda t: phone_re.sub(space_pad(metatoken_telno), t),
    lambda t: cc_phone_re.sub(space_pad(metatoken_cctelno + " " + metatoken_telno), t),
    lambda t: date_1_re.sub(space_pad(metatoken_ddmmyyyy), t),
    lambda t: date_2_re.sub(space_pad(metatoken_ddmmyyyy), t),
    lambda t: time_re.sub(space_pad(metatoken_hhmm), t),
    lambda t: carplate_re.sub(space_pad(metatoken_plate), t),

    lambda t: space.sub(" ", t)
)

prefix_re = re.compile('^[-]+')


def tokenize_substring(text: str):
    for t in token_re.findall(text):
        yield prefix_re.sub('', t)


def word_tokenize_iter(text: str):
    for substring in c(text).split(' '):
        for t in tokenize_substring(substring):
            if len(t) > 0:
                yield t


def word_tokenize(text: str):
    return list(word_tokenize_iter(text))


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
