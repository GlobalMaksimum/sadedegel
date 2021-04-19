import pytest
from .context import hotel_sentiment, CLASS_VALUES


def test_model_load():
    _ = hotel_sentiment.load()


def test_inference():
    model = hotel_sentiment.load()

    pred = model.predict(['asla gidilmeyecek bir otel hasta oldukotel tam anlamıyla bir fiyasko satın alırken ve web '
                          'sitesinde gözünüze çarpan en büyük özellik otelin tüm alanlarının yenilenmiş olması ama '
                          'bunun gerçekle alakası yok odalar en az  yıllık bir otel harabeliğinde yemekler ve '
                          'özellikle kahvaltı tam bir hayal kırıklığı kahvaltıdaki yiyecekler asla yenmeyecek ve '
                          'yedirilmeyecek kadar kötü bir tane lekesiz temiz bir tabak bardak veya çatal kaşık '
                          'görmeniz olası bile değil içecek konusunda su değil zehirli su katılmış gibi gerçeği ile '
                          'alakası olmayan içecekler ıce tea yada soğuk çay cinsi birşey otelde asla yok konsepte '
                          'uygun değilmiş açıklama bu soğuk çay hangi konseptin ki acaba bu otele uymuyor garsonların '
                          'hepsi kendi dalında bir kabadayı restaurant müdürü denen kişi inanılmaz yeteneksiz asla '
                          'yeme içme kültürü yok gerçekten birsey isteyipte almanız mucize birde asıl bir mevzu varki '
                          'anlatılmaz bu otelde can güvenliğiniz yok oda anahtarı kardeşimdeydi ben resepsiyona '
                          'anahtar almaya gittim sırf anahtar yapmamak için elimde anahtar yok dedi benim sorunum '
                          'değil bulacaksınız ben odama gireceğim anahtar kardeşimde oda otel dışında dedim sordu oda '
                          'numaramı yaptı verdi ama tuhaf olan şuki ne oda numaramdan adımı kontrol etti yada '
                          'hiçbirşey sormadı bizi daha öncede görmedi ki güven esaslı verdi diyeceğim yani herkes oda '
                          'anahtarını alıp herseyi yapabilir otelde şampuan yok tamam kimse kullanmıyor belki ama * '
                          'lı bir otelde nasıl olmaz otelde terlik yok yani yoklar oteli ama şunu söylemem gereki ki '
                          'housekeeping deki çalışanlar çok iyi hk yöneticileri asla insana değer vermeyen asık '
                          'suratlı insanlar tatil dönüşü kendimi kardeşimle beraber hastanede bulduk tatil boyunca '
                          'azıcıkda olsa yediğimiz herseyi çıkardık ve geldiğimizde serum alacak kadar hasta olduk '
                          'biz gittiğimizde otelin sahibide oteldeydi tüm şikayetleri memnuniyetsizlikleri duyuyor '
                          'ama asla umurlarında olmuyor sahili çok kötü kıyısı berrak değil iskele dökülüyor asla '
                          'gidilmeyecek bir otel',
                          'Çok güzeldi 4 cü gidişimiz tesise cok memnun kaldık.'])

    assert CLASS_VALUES[pred[0]] in CLASS_VALUES
    assert CLASS_VALUES[pred[1]] in CLASS_VALUES

    probability = model.predict_proba(
        ['asla gidilmeyecek bir otel hasta oldukotel tam anlamıyla bir fiyasko satın alırken ve web '
         'sitesinde gözünüze çarpan en büyük özellik otelin tüm alanlarının yenilenmiş olması ama '
         'bunun gerçekle alakası yok odalar en az  yıllık bir otel harabeliğinde yemekler ve '
         'özellikle kahvaltı tam bir hayal kırıklığı kahvaltıdaki yiyecekler asla yenmeyecek ve '
         'yedirilmeyecek kadar kötü bir tane lekesiz temiz bir tabak bardak veya çatal kaşık '
         'görmeniz olası bile değil içecek konusunda su değil zehirli su katılmış gibi gerçeği ile '
         'alakası olmayan içecekler ıce tea yada soğuk çay cinsi birşey otelde asla yok konsepte '
         'uygun değilmiş açıklama bu soğuk çay hangi konseptin ki acaba bu otele uymuyor garsonların '
         'hepsi kendi dalında bir kabadayı restaurant müdürü denen kişi inanılmaz yeteneksiz asla '
         'yeme içme kültürü yok gerçekten birsey isteyipte almanız mucize birde asıl bir mevzu varki '
         'anlatılmaz bu otelde can güvenliğiniz yok oda anahtarı kardeşimdeydi ben resepsiyona '
         'anahtar almaya gittim sırf anahtar yapmamak için elimde anahtar yok dedi benim sorunum '
         'değil bulacaksınız ben odama gireceğim anahtar kardeşimde oda otel dışında dedim sordu oda '
         'numaramı yaptı verdi ama tuhaf olan şuki ne oda numaramdan adımı kontrol etti yada '
         'hiçbirşey sormadı bizi daha öncede görmedi ki güven esaslı verdi diyeceğim yani herkes oda '
         'anahtarını alıp herseyi yapabilir otelde şampuan yok tamam kimse kullanmıyor belki ama * '
         'lı bir otelde nasıl olmaz otelde terlik yok yani yoklar oteli ama şunu söylemem gereki ki '
         'housekeeping deki çalışanlar çok iyi hk yöneticileri asla insana değer vermeyen asık '
         'suratlı insanlar tatil dönüşü kendimi kardeşimle beraber hastanede bulduk tatil boyunca '
         'azıcıkda olsa yediğimiz herseyi çıkardık ve geldiğimizde serum alacak kadar hasta olduk '
         'biz gittiğimizde otelin sahibide oteldeydi tüm şikayetleri memnuniyetsizlikleri duyuyor '
         'ama asla umurlarında olmuyor sahili çok kötü kıyısı berrak değil iskele dökülüyor asla '
         'gidilmeyecek bir otel',
         'Çok güzeldi 4 cü gidişimiz tesise cok memnun kaldık.'])

    assert probability.shape == (2, 2)
