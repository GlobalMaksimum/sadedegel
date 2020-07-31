from .context import RegexpSentenceTokenizer, NLTKPunctTokenizer, Doc

sentences = [("7 mucize bitki 18 Şubat 2018 PAYLAŞ Yorum yaz A Bağışıklık sistemini uyaran tıbbi bitkiler, fiziksel,"
              " kimyasal, biyolojik ve psikolojik streslere karşı vücut direncini artırıyor."),
             "Antioksidan koruma sistemini yeniliyor.",
             " Bağışıklık sisteminin düzgün çalışabilmesi için sağlıklı bir bağırsak florası oluşmasına destek oluyor.",
             (
                 " Ekinezya, Kore ginsengi, çörek otu, zerdeçal (Hint safranı), zencefil, sarımsak, Güney Afrika "
                 "sardunyasını ‘mucize bitkiler’ olarak niteleyen geleneksel ve tamamlayıcı tıp hekimi, fitoterapist "
                 "Dr. Serdar Özgüç, tüm tıbbi biktilerin hekim kontrolünde kullanılması gerektiği uyarısında bulunarak,"
                 " şu bilgileri veriyor:"),
             "EKİNEZYA: Asırlardır Kızılderili bitkisi olarak biliniyor.",
             " ‘Soğuk algınlığı savaşçısı’ olarak Avrupa ve Amerika’da gıda takviyesi olarak kullanılıyor."
             ]

text = " ".join(sentences)


def test_punct_tokenizer():
    toker = NLTKPunctTokenizer()

    pred, true = toker(text), map(lambda x: x.strip(), sentences)

    assert len(set(pred) & set(true)) / len(set(pred) | set(true)) >= 0.5


def test_re_tokenizer():
    toker = RegexpSentenceTokenizer()

    pred, true = toker(text), map(lambda x: x.strip(), sentences)

    assert len(set(pred) & set(true)) / len(set(pred) | set(true)) >= 0.5


def test_ml_sent_tokenizer_edge_cases():
    doc_str = ("Hz. İsa M.Ö. 0. yılda doğdu. Doç. Dr. Mehmet Bey kanserin ilacını buldu!!! Aşk… "
               "14 Şubat'ta olmasın… Kocatepe Mah.de, Güneş Sok.ta gerçekleşen olay herkesi şaşırttı! "
               "Alb. Gen. Mehmet Bey kendi evine saldırı düzenledi... "
               "K.K.T.C'de eşek nüfusu kontrol dışında! "
               "Av. İst. Prof. Mehmet Bey Tahtalıköy'e uğradı. "
               "123. Türkiye E-Sports turnuvası İstanbul'da gerçekleşti.")

    doc = Doc(doc_str)

    assert [sent.text for sent in doc.sents] == ["Hz. İsa M.Ö. 0. yılda doğdu.",
                                                 'Doç. Dr. Mehmet Bey kanserin ilacını buldu!!!', 'Aşk…',
                                                 "14 Şubat'ta olmasın…",
                                                 "Kocatepe Mah.de, Güneş Sok.ta gerçekleşen olay herkesi şaşırttı!",
                                                 "Alb. Gen. Mehmet Bey kendi evine saldırı düzenledi...",
                                                 "K.K.T.C'de eşek nüfusu kontrol dışında!",
                                                 "Av. İst. Prof. Mehmet Bey Tahtalıköy'e uğradı.",
                                                 "123. Türkiye E-Sports turnuvası İstanbul'da gerçekleşti."]


def building_doc_using_sents():
    sents = ["Hz. İsa M.Ö. 0. yılda doğdu.",
             'Doç. Dr. Mehmet Bey kanserin ilacını buldu!!!', 'Aşk…',
             "14 Şubat'ta olmasın…",
             "Kocatepe Mah.de, Güneş Sok.ta gerçekleşen olay herkesi şaşırttı!",
             "Alb. Gen. Mehmet Bey kendi evine saldırı düzenledi...",
             "K.K.T.C'de eşek nüfusu kontrol dışında!",
             "Av. İst. Prof. Mehmet Bey Tahtalıköy'e uğradı.",
             "123. Türkiye E-Sports turnuvası İstanbul'da gerçekleşti."]

    doc = Doc(None, sents)

    assert [sent.text for sent in doc.sents] == sents
