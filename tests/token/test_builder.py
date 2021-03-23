from .context import VocabularyCounter, Token


def test_builder():
    v = VocabularyCounter("bert", case_sensitive=True)

    docs = [['Yaradılmış', 'cümle', 'oldu', 'şaduman'], ['Gam', 'gidip', 'alem', 'yeniden', 'buldu', 'can'],
            ['Cümle', 'zerrat-ı', 'cihan', 'idüb', 'nida']]

    for i, words in enumerate(docs):
        for word in words:
            v.add_word_to_doc(word, i)

    assert v.df("cümle") == 1 and v.df("Cümle") == 1

    assert v.vocabulary_size == 15
    assert v.document_count == 3


def test_singleton():
    assert id(Token("Yaradılmış")) == id(Token("Yaradılmış"))
