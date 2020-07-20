from .context import tr_upper, tr_lower, __tr_upper__, __tr_lower__, load_raw_corpus, Doc, flatten


def test_istitle():
    assert "İstanbul".istitle()


def test_isupper():
    assert __tr_upper__.isupper()


def test_islower():
    assert __tr_lower__.islower()


def test_lower():
    assert tr_lower(__tr_upper__) == __tr_lower__


def test_lower2():
    assert tr_lower(__tr_lower__) == __tr_lower__


def test_upper():
    assert tr_upper(__tr_lower__) == __tr_upper__


def test_upper2():
    assert tr_upper(__tr_upper__) == __tr_upper__


def test_isdigit():
    assert '0123456789'.isdigit()


def test_span_feature_generation():
    raw_corpus = load_raw_corpus()

    _ = [[span.span_features() for span in Doc(raw).spans] for raw in raw_corpus]


def test_flattening_span_features():
    raw_corpus = load_raw_corpus()

    _ = flatten([[span.span_features() for span in Doc(raw).spans] for raw in raw_corpus])
