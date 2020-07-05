from .context import RegexpSentenceTokenizer, NLTKPunctTokenizer

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

text = "".join(sentences)


def test_punct_tokenizer():
    toker = NLTKPunctTokenizer()

    pred, true = toker(text), map(lambda x: x.strip(), sentences)

    assert len(set(pred) & set(true)) / len(set(pred) | set(true)) >= 0.5


def test_re_tokenizer():
    toker = RegexpSentenceTokenizer()

    pred, true = toker(text), map(lambda x: x.strip(), sentences)

    assert len(set(pred) & set(true)) / len(set(pred) | set(true)) >= 0.5
