---
short_name: datasets
id: juniper-datasets
code: |
    from sadedegel.dataset import load_raw_corpus
    raw = load_raw_corpus()
    d = next(raw)
    d
---

## SadedeGel Veri Kümeleri

SadedeGel ile birlikte farklı formatlarda hazır veri kümeleri gelmektedir. Bunların bazıları ham veriler iken bazıları human-annotated veri kümeleridir.
