from .context import Doc


def test_binary_tf():
    d = Doc(
        "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

    binary = d[0].get_tf("binary")

    assert binary[13516] == 1
    assert binary[13490] == 1
    assert binary[3] == 1
    assert binary[5686] == 1
    assert binary[6968] == 1
    assert binary[193] == 1
    assert binary[13553] == 1
    assert binary[33] == 1
    assert binary[89] == 1

    assert binary.sum() == 13


def test_raw_tf():
    d = Doc(
        "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

    raw = d[0].get_tf("raw")

    assert raw[13516] == 1
    assert raw[13490] == 1
    assert raw[3] == 1
    assert raw[5686] == 1
    assert raw[6968] == 1
    assert raw[193] == 1
    assert raw[13553] == 1
    assert raw[33] == 1
    assert raw[89] == 2

    assert raw.sum() == 14


def test_raw_tf_without_stopwords():
    d = Doc(
        "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

    raw = d[0].get_tf("raw", drop_stopwords=True)

    assert raw[13516] == 1
    assert raw[13490] == 1
    assert raw[3] == 1
    assert raw[5686] == 1
    assert raw[6968] == 1
    assert raw[193] == 1
    assert raw[13553] == 1

    assert raw.sum() == 10


def test_raw_tf_without_stopwords_lowercase():
    d = Doc(
        "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev bir alan bir olmadı.")

    raw = d[0].get_tf("raw", drop_stopwords=True, lowercase=True)

    assert raw[22781] == 1
    assert raw[13490] == 1
    assert raw[3] == 1
    assert raw[5686] == 1
    assert raw[6968] == 1
    assert raw[193] == 1
    assert raw[13553] == 1

    assert raw.sum() == 10


def test_raw_tf_lowercase():
    d = Doc(
        "Bilişim sektörü, günlük devrimlerin yaşandığı ve dev Bir alan bir olmadı.")

    raw = d[0].get_tf("raw", lowercase=True)

    assert raw[22781] == 1
    assert raw[13490] == 1
    assert raw[3] == 1
    assert raw[5686] == 1
    assert raw[6968] == 1
    assert raw[193] == 1
    assert raw[13553] == 1
    assert raw[89] == 2

    assert raw.sum() == 14
