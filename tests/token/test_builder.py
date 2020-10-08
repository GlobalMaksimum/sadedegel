from .context import Vocabulary


def test_builder():
    Vocabulary.vocabularies.clear()
    vocabulary = Vocabulary.factory("bert")

    docs = [['Yaradılmış', 'cümle', 'oldu', 'şaduman'], ['Gam', 'gidip', 'alem', 'yeniden', 'buldu', 'can'],
            ['Cümle', 'zerrat-ı', 'cihan', 'idüb', 'nida']]

    for i, words in enumerate(docs):
        for word in words:
            vocabulary.add_word_to_doc(word, i)

    vocabulary.build(1)

    entity = vocabulary['cümle']

    assert entity.df == 2 and entity.df_cs == 1

    entity = vocabulary['Cümle']

    assert entity.df == 2 and entity.df_cs == 1

    assert len(vocabulary) == 15
    assert vocabulary.document_count == 3


def test_builder_with_df_2():
    vocabulary = Vocabulary.factory("mert")

    docs = [['Yaradılmış', 'cümle', 'oldu', 'şaduman'], ['Gam', 'gidip', 'alem', 'yeniden', 'buldu', 'can'],
            ['Cümle', 'zerrat-ı', 'cihan', 'idüb', 'nida']]

    for i, words in enumerate(docs):
        for word in words:
            vocabulary.add_word_to_doc(word, i)

    vocabulary.build(2)

    entity = vocabulary['cümle']

    assert entity.df == 2 and entity.df_cs == 1

    entity = vocabulary['Cümle']

    assert entity.df == 2 and entity.df_cs == 1

    assert len(vocabulary) == 2
    assert vocabulary.document_count == 3
