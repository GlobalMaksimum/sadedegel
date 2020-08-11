from fastapi.testclient import TestClient
import pytest
from itertools import product
from .context import app

client = TestClient(app)


def test_random_summarizer_req():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.", wpm=1,
        duration=1))
    assert response.status_code == 200
    assert 'sentences' in response.json()


def test_firstk_summarizer_req():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.", wpm=1,
        duration=1))
    assert response.status_code == 200
    assert 'sentences' in response.json()


def test_random_summarizer_wpm0():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.", wpm=0,
        duration=1))

    assert response.status_code == 200
    assert 'sentences' in response.json()


def test_random_summarizer_duration0():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.", wpm=150,
        duration=0))

    assert response.status_code == 200
    assert 'sentences' in response.json()


def test_random_summarizer_wpm0_duration0():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi! Mahmut Hoca'nın emriyle Uganda Cumhurbaşkanı'nı karşılamaya gidiyoruz.", wpm=0,
        duration=0))

    assert response.status_code == 200
    assert 'sentences' in response.json()


def test_random_summarizer_nosentence():
    response = client.post("/api/summarizer/random", json=dict(
        doc="Kapıyı aç Veysel Efendi", wpm=0,
        duration=0))

    assert response.status_code == 200
    assert 'sentences' in response.json()
    assert len(response.json()['sentences']) == 1


@pytest.mark.parametrize("summarizer, ", ['random', 'rouge1', 'firstk'])
def test_in_order(summarizer):
    doc = "aa aa aa. bb bb cc. aa bb."
    sents = ['aa aa aa.', 'bb bb cc.', 'aa bb.']

    response = client.post(f"/api/summarizer/{summarizer}", json=dict(
        doc=doc, wpm=4,
        duration=3))

    assert response.status_code == 200 and 'sentences' in response.json()

    indexes = [sents.index(s) for s in response.json()['sentences']]

    assert indexes == sorted(indexes)


testdata = [('/api/summarizer/random', 'https://www.hurriyet.com.tr', 'POST'),
            ('/', 'https://www.hurriyet.com.tr', 'GET'),
            ('/api/summarizer/random', 'http://0.0.0.0:8000/sadedegel', 'POST'),
            ('/api/summarizer/random', 'https://www.milliyet.com.tr', 'GET'),
            ('/api/summarizer/random', 'https://www.sozcu.com.tr', 'GET')]


@pytest.mark.parametrize("url, origin, method", testdata)
def test_CORS(url, origin, method):
    response = client.options(url, headers={"Origin": origin,
                                            'Access-Control-Request-Method': method})
    assert response.status_code == 200
