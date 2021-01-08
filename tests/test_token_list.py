from .context import Doc, Token


famous_quote = "Akın Gökyürüyen karanlık işlere bulaşmıştı. Akıl hocasını dinlemedi. Sevdasının peşinde çürüdü."


def test_list():
    d = Doc(famous_quote)
    assert isinstance(d[0][0], Token)
    assert d[1][0].word == "Akıl"


def test_iter():
    d = Doc(famous_quote)
    for s, _ in enumerate(d):
        for i, tok in enumerate(d[s]):
            assert d[s].tokens[i] == tok.word
            assert isinstance(tok, Token)
