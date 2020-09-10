---
short_name: extraction-summarizers
title: SadedeGel Extraction Based Özetleyiciler
id: juniper-summarizers
code: |
    import numpy as np
    from sadedegel.tokenize import Doc
    from sadedegel.dataset import load_raw_corpus
    from sadedegel.summarize import Rouge1Summarizer

    raw = load_raw_corpus(return_iter=False)

    d = Doc(raw[0])

    print(f"Metin içerisindeki toplam cümle sayısı {len(d.sents)}")
    print()

    scores = Rouge1Summarizer().predict(d.sents)

    print(scores)
    print()

    top3_index = np.argsort(scores)[::-1][:3]

    for sent in np.array(d.sents)[top3_index]:
        print(f'⇨ {sent}')
---

## SadedeGel Extraction Based Özetleyiciler

SadedeGel extraction-based özetleyicilerin tamamı, temelde cümle skorlaması yapan kural veya ML tabanlı sınıflardır.
