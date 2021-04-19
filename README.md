<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/assets/img/logo-2.png" width="125" height="125" align="right" /></a>

# SadedeGel: A General Purpose NLP library for Turkish

SadedeGel is initially designed to be a library for unsupervised extraction-based news summarization using several old and new NLP techniques.

Development of the library started as a part of [Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/)

We keep adding lots to become a general purpose open source NLP library for Turkish langauge.


💫 **Version 0.20 out now!**
[Check out the release notes here.](https://github.com/GlobalMaksimum/sadedegel/releases)


![Python package](https://github.com/GlobalMaksimum/sadedegel/workflows/Python%20package/badge.svg)
[![Python Version](https://img.shields.io/pypi/pyversions/sadedegel?style=plastic)](https://img.shields.io/pypi/pyversions/sadedegel)
[![Coverage](https://codecov.io/gh/globalmaksimum/sadedegel/branch/master/graphs/badge.svg?style=plastic)](https://codecov.io/gh/globalmaksimum/sadedegel)
[![pypi Version](https://img.shields.io/pypi/v/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![PyPi downloads](https://img.shields.io/pypi/dm/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![License](https://img.shields.io/pypi/l/sadedegel)](https://github.com/GlobalMaksimum/sadedegel/blob/master/LICENSE)
![Commit Month](https://img.shields.io/github/commit-activity/m/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
![Commit Week](https://img.shields.io/github/commit-activity/w/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
![Last Commit](https://img.shields.io/github/last-commit/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
[![Binder](https://mybinder.org/badge_logo.svg?style=plastic)](https://mybinder.org/v2/gh/GlobalMaksimum/sadedegel.git/master?filepath=notebook%2FBasics.ipynb)
[![Slack](https://img.shields.io/static/v1?logo=slack&style=plastic&color=blueviolet&label=slack&labelColor=grey&message=sadedegel)](https://join.slack.com/t/sadedegel/shared_invite/zt-h77u6aeq-VzEorB5QLHyJV90Fv4Ky3A)
[![Kaggle](http://img.shields.io/static/v1?logo=kaggle&style=plastic&color=blue&label=kaggle&labelColor=grey&message=notebooks)](https://www.kaggle.com/search?q=sadedegel+in%3Anotebooks)


## 📖 Documentation

| Documentation   |                                                                |
| --------------- | -------------------------------------------------------------- |
| [Contribute]    | How to contribute to the sadedeGel project and code base.          |

[contribute]: https://github.com/GlobalMaksimum/sadedegel/blob/master/CONTRIBUTING.md

## 💬 Where to ask questions

The SadedeGel project is initialized by [@globalmaksimum](https://github.com/GlobalMaksimum) AI team members
[@dafajon](https://github.com/dafajon),
[@askarbozcan](https://github.com/askarbozcan),
[@mccakir](https://github.com/mccakir),
[@husnusensoy](https://github.com/husnusensoy) and 
[@ertugruldemir](https://github.com/ertugrul-dmr).


Other community maintainers

* [@doruktiktiklar](https://github.com/doruktiktiklar) contributes [TFIDF Summarizer](sadedegel/summarize/tf_idf.py)

| Type                     | Platforms                                              |
| ------------------------ | ------------------------------------------------------ |
| 🚨 **Bug Reports**       | [GitHub Issue Tracker]                                 |
| 🎁 **Feature Requests**  | [GitHub Issue Tracker]                                 |
| <img width="18" height="18" src="https://www.freeiconspng.com/uploads/slack-icon-2.png"/> **Questions**  | [Slack Workspace]                                 |

[github issue tracker]: https://github.com/GlobalMaksimum/sadedegel/issues
[Slack Workspace]: https://join.slack.com/t/sadedegel/shared_invite/zt-h77u6aeq-VzEorB5QLHyJV90Fv4Ky3A


## Features

* Several datasets
  * Basic corpus
      * Raw corpus (`sadedegel.dataset.load_raw_corpus`)
      * Sentences tokenized corpus (`sadedegel.dataset.load_sentences_corpus`)  
      * Human annotated summary corpus (`sadedegel.dataset.load_annotated_corpus`)   
  * [Extended corpus](sadedegel/dataset/README.md)
      * Raw corpus (`sadedegel.dataset.extended.load_extended_raw_corpus`)
      * Sentences tokenized corpus (`sadedegel.dataset.extended.load_extended_sents_corpus`)
      
  * TsCorpus(`sadedegel.dataset.tscorpus`)
      * Thanks to [Taner Sezer](https://github.com/tanerim), over 300K documents from tscorpus is also a part of sadedegel. Allowing us to
        * [Evaluate](sadedegel/bblock/TOKENIZER.md) our tokenizers (word tokenizers)
        * Build our [prebuilt news category classifier](sadedegel/prebuilt/README.md)  
* ML based sentence boundary detector (**SBD**) trained for Turkish language (`sadedegel.dataset`)
* Sadedegel Extractive Summarizers
  * Various baseline summarizers
    * Position Summarizer
    * Length Summarizer
    * Band Summarizer
    * Random Summarizer
  
  * Various unsupervised/supervised summarizers
    * ROUGE1 Summarizer
    * TextRank Summarizer
    * Cluster Summarizer
    * Lexrank Summarizer
    * BM25 Summarizer
    * TfIdf Summarizer
 
* Various Word Tokenizers
  * BERT Tokenizer - Trained tokenizer (`pip install sadedegel[bert]`)
  * Simple Tokenizer - Regex Based
  * IcU Tokenizer (default by `0.19`)
  
* Various Embeddings Implementation
  * BERT Embeddings (`pip install sadedegel[bert]`)
  * TfIdf Embeddings
  
* [**Experimental**] Prebuilt models for several common NLP tasks ([`sadedegel.prebuilt`](sadedegel/prebuilt/README.md)).

```python
from sadedegel.prebuilt import news_classification

model = news_classification.load()

doc_str = ("Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı. Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok "
"daha büyük ölçekte. Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu. IBM 650 Model I adını taşıyan bilgisayarın "
"satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı. Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı. "
"Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.")

y_pred = model.predict([doc_str])
```

📖 **For more details, refer to [sadedegel.ai](http://sadedegel.ai)**

## Install sadedeGel

- **Operating system**: macOS / OS X · Linux · Windows (Cygwin, MinGW, Visual
  Studio)
- **Python version**: 3.6+ (only 64 bit)
- **Package managers**: [pip] 

[pip]: https://pypi.org/project/sadedegel/

### pip

Using pip, sadedeGel releases are available as source packages and binary wheels.

```bash
pip install sadedegel
```
or update now

```bash
pip install sadedegel -U
```

When using pip it is generally recommended to install packages in a virtual
environment to avoid modifying system state:

```bash
python -m venv .env
source .env/bin/activate
pip install sadedegel
```

#### Optional

To keep core sadedegel as light as possible we decomposed our initial monolitic design.

To enable BERT embeddings and related capabilities use

```bash
pip install sadedegel[bert]
```

We ship 100-dimension word vectors with the library. If you need to retrain those embeddings you can use

```bash
python -m sadedegel.bblock.cli build-vocabulary
```

`--w2v` option requires `w2v` option to be installed. To install option use

```bash
pip install sadedegel[w2v]
```

### Quickstart with SadedeGel

To load SadedeGel, use `sadedegel.load()`

```python
from sadedegel import Doc
from sadedegel.dataset import load_raw_corpus
from sadedegel.summarize import Rouge1Summarizer

raw = load_raw_corpus()

d = Doc(next(raw))

summarizer = Rouge1Summarizer()
summarizer(d, k=5)
```

To trigger sadedeGel NLP pipeline, initialize `Doc` instance with a document string.

Access all sentences using Python built-in `list` function.

```python
from sadedegel import Doc

doc_str = ("Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı. Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok "
"daha büyük ölçekte. Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu. IBM 650 Model I adını taşıyan bilgisayarın "
"satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı. Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı. "
"Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.")

doc = Doc(doc_str)

list(doc)
```
```python
['Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı.',
 'Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok daha büyük ölçekte.',
 'Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu.',
 'IBM 650 Model I adını taşıyan bilgisayarın satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı.',
 'Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı.',
 'Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.']
```

Access sentences by index.

```python
doc[2]
```

```python
Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu.
```

## SadedeGel Server
In order to integrate with your applications we provide a quick summarizer server with sadedeGel.

```bash
python3 -m sadedegel.server 
```

### SadedeGel Server on Heroku
[SadedeGel Server](https://sadedegel.herokuapp.com/api/info) is hosted on free tier of [Heroku](https://heroku.com) cloud services.

* [OpenAPI Documentation](https://sadedegel.herokuapp.com/docs)
* [Redoc Documentation](https://sadedegel.herokuapp.com/redoc)
* [Redirection to sadedegel.ai](https://sadedegel.herokuapp.com)

## PyLint, Flake8 and Bandit
sadedeGel utilized [pylint](https://www.pylint.org/) for static code analysis, 
[flake8](https://flake8.pycqa.org/en/latest) for code styling and [bandit](https://pypi.org/project/bandit) 
for code security check.

To run all tests

```bash
make lint
```

## Run tests

sadedeGel comes with an [extensive test suite](sadedegel/tests). In order to run the
tests, you'll usually want to clone the repository and build sadedeGel from source.
This will also install the required development dependencies and test utilities
defined in the `requirements.txt`.

Alternatively, you can find out where sadedeGel is installed and run `pytest` on
that directory. Don't forget to also install the test utilities via sadedeGel's
`requirements.txt`:

```bash
make test
```

## 📓 Kaggle

* Check [comprehensive notebook](https://www.kaggle.com/datafan07/clickbait-news-classification-using-sadedegel) of Kaggle Master [Ertugrul Demir](https://www.kaggle.com/datafan07) explaining the capabilities of sadedegel on Turkish clickbate dataset


## Youtube Channel
Some videos from [sadedeGel YouTube Channel](https://www.youtube.com/channel/UCyNG1Mehl44XWZ8LzkColuw)

### SkyLab YTU Webinar Playlist

[![Youtube](https://img.shields.io/youtube/likes/xoEERspk6Is?label=SadedeGel%20Subprojects%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=xoEERspk6Is)

[![Youtube](https://img.shields.io/youtube/likes/HfWIzAwf5u8?label=SadedeGel%20Scraper%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=HfWIzAwf5u8)

[![Youtube](https://img.shields.io/youtube/likes/PkUmYhahiMw?label=SadedeGel%20Evaluation-nDCG%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=PkUmYhahiMw)

[![Youtube](https://img.shields.io/youtube/likes/AxpK7fOndRQ?label=SadedeGel%20Annotator%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=AxpK7fOndRQ)

[![Youtube](https://img.shields.io/youtube/likes/jKh_t9ZOJ-g?label=SadedeGel%20Baseline%20Özetleyiciler%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=jKh_t9ZOJ-g)

[![Youtube](https://img.shields.io/youtube/likes/3DO1X7de1FI?label=SadedeGel%20ROUGE1%20Özetleyici%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=3DO1X7de1FI)

[![Youtube](https://img.shields.io/youtube/likes/KGg3DJQVH9c?label=SadedeGel%20Kümeleme%20Bazlı%20Özetleyiciler%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=KGg3DJQVH9c)

[![Youtube](https://img.shields.io/youtube/likes/G_erifsGGFs?label=SadedeGel%20BERT%20Embeddings%20(Turkish)&style=social&withDislikes)](https://www.youtube.com/watch?v=G_erifsGGFs)

## References

### Special Thanks

* [Starlang Software](https://starlangyazilim.com/) for their contribution to open source Turkish NLP development and corpus preperation.

* [Olcay Taner Yıldız, Ph.D.](https://github.com/olcaytaner), one of our refrees in [Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/), for helping our development on sadedegel.

* [Taner Sezer](https://github.com/tanerim) for his contribution on tokenization corpus and labeled news corpus.

### Our Community Contributors

We would like to thank our community contributors for their bug/enhancement requests and questions to make sadedeGel better everyday

* [Burak Işıklı](https://github.com/burakisikli)

### Software Engineering
* Special thanks to [spaCy](https://github.com/explosion/spaCy) project for their work in showing us the way to implement a proper python module rather than merely explaining it.
    * We have borrowed many document and style related stuff from their code base :smile:
    
* There are a few free-tier service providers we need to thank:
  * [GitHub](https://github.com) for
      * Hosting our projects.
      * Making it possible to collobrate easily.
      * Automating our SLM via [Github Actions](https://github.com/features/actions)
  * [Google Cloud Google Storage Service](https://cloud.google.com/products/storage) for providing low cost storage buckets making it possible to store `sadedegel.dataset.extended` data.
  * [Heroku](https://heroku.com) for hosting [sadedeGel Server](https://sadedegel.herokuapp.com/api/info) in their free tier dynos.
  * [CodeCov](https://codecov.io/) for allowing us to transparently share our [test coverage](https://codecov.io/gh/globalmaksimum/sadedegel)
  * [PyPI](https://pypi.org/) for allowing us to share [sadedegel](https://pypi.org/project/sadedegel) with you.
  * [binder](https://mybinder.org/) for 
     * Allowing us to share our example [notebooks](notebook/)
     * Hosting our learn by example boxes in [sadedegel.ai](http://sadedegel.ai) 
    
### Machine Learning (ML), Deep Learning (DL) and Natural Language Processing (NLP)
* Resources on Extractive Text Summarization:

    * [Leveraging BERT for Extractive Text Summarization on Lectures](https://arxiv.org/abs/1906.04165)  by Derek Miller
    * [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf) by Yang Liu

* Other NLP related references

    * [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013.pdf)
    * [Speech and Language Processing, Second Edition](https://web.stanford.edu/~jurafsky/slp3/)
