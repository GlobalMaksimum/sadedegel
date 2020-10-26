from typing import List
import numpy as np
import warnings

__tr_upper__ = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
__tr_lower__ = "abcçdefgğhıijklmnoöprsştuüvyz"

__tr_lower_abbrv__ = ['hz.', 'dr.', 'prof.', 'doç.', 'org.', 'sn.', 'st.', 'mah.', 'mh.', 'sok.', 'sk.', 'alb.', 'gen.',
                      'av.', 'ist.', 'ank.', 'izm.', 'm.ö.', 'k.k.t.c.']


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").upper()


def space_pad(token):
    return " " + token + " "


def space_pad(token):
    return " " + token + " "


def pad(l, padded_length):
    return l + [0 for _ in range(padded_length - len(l))]


def flatten(l2: List[List]):
    flat = []
    for l in l2:
        for e in l:
            flat.append(e)

    return flat


def is_eos(span, sentences: List[str]) -> int:
    start = 0
    eos = []
    for s in sentences:
        idx = span.doc.raw.find(s, start) + len(s) - 1
        eos.append(idx)

        start = idx

    b, e = span.value

    for idx in eos:
        if b <= idx <= e:
            return 1

    return 0


def normalize_tokenizer_name(tokenizer_name, raise_on_error=False):
    normalized = tokenizer_name.lower().replace(' ', '').replace('-', '').replace('tokenizer', '')

    if normalized not in ['bert', 'simple']:
        msg = f"Invalid tokenizer {tokenizer_name} ({normalized}). Valid values are bert, simple"
        if raise_on_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=3)

    return normalized
