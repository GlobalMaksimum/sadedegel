from .context import Doc, Vocabulary


def test_oov():
    d = Doc(("o macta adam 6 7 tane net sut cikartti aksini soleyen ya maldir yada futboldan anlamiyodur "
             "buna benzer bir pozisyon da wesley uzaktan attigi golun birinde de ayni pozisyon vardi ozaman bunu "
             "size en iyi volkan aciklar kudurun fesat ibneler."))

    embedding = d.tfidf_embeddings
