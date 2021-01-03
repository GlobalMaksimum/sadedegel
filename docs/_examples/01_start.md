---
short_name: start
title: Sadedegel'e Hızlı Giriş
id: juniper-intro
code: |
    from sadedegel.tokenize import Doc
    text = """
    Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.
    """

    d = Doc(text)

    print(d.sents)
    print()
    print(f"Cümle 1 - Rouge1: {d.sents[0].rouge1('recall')} (recall) {d.sents[0].rouge1('precision')} (precision)")
    print()

    print(f"Cümle 1 uzunluğu: {len(d.sents[0])} ")
    print(d.sents[0].tokens)
---

## Sadedegel'e Hızlı Giriş

Sadedegel kütüphanesinde bir çok akış Doc sınıfıyla başlar.
Bir metin ile Doc objesini çağırdığınızda

-   Sentence Boundary Detection (SBD)
-   Her bir cümle için word tokenization
-   Cümle BERT embedding hesaplaması
-   Cümle'ye ait rouge1 score hesaplaması

gibi işlemler gerçekleşir.
