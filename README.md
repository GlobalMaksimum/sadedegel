<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/dist/img/logo-2.png?s=280&v=4" width="125" height="125" align="right" /></a>

# SadedeGel: An extraction based Turkish news summarizer

SadedeGel is a library for unsupervised extraction-based news summarization using several old and new NLP techniques.

Development of the library takes place as a part of [Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/)

💫 **Version 0.13 out now!**
[Check out the release notes here.](https://github.com/GlobalMaksimum/sadedegel/releases)


![Python package](https://github.com/GlobalMaksimum/sadedegel/workflows/Python%20package/badge.svg)
[![Python Version](https://img.shields.io/pypi/pyversions/sadedegel?style=plastic)](https://img.shields.io/pypi/pyversions/sadedegel)
[![Coverage](https://codecov.io/gh/globalmaksimum/sadedegel/branch/master/graphs/badge.svg?style=plastic)](https://codecov.io/gh/globalmaksimum/sadedegel)
[![Code Quality Score](https://www.code-inspector.com/project/12884/score/svg?style=plastic)](https://frontend.code-inspector.com/public/project/12884/sadedegel/dashboard)
[![Code Grade](https://www.code-inspector.com/project/12884/status/svg?style=plastic)](https://frontend.code-inspector.com/public/project/12884/sadedegel/dashboard)
[![pypi Version](https://img.shields.io/pypi/v/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![PyPi downloads](https://img.shields.io/pypi/dm/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![License](https://img.shields.io/pypi/l/sadedegel)](https://github.com/GlobalMaksimum/sadedegel/blob/master/LICENSE)
![Commit Month](https://img.shields.io/github/commit-activity/m/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
![Commit Week](https://img.shields.io/github/commit-activity/w/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
![Last Commit](https://img.shields.io/github/last-commit/globalmaksimum/sadedegel?style=plastic&logo=GitHub)
[![Binder](https://mybinder.org/badge_logo.svg?style=plastic)](https://mybinder.org/v2/gh/GlobalMaksimum/sadedegel.git/master?filepath=notebook%2FBasics.ipynb)


## 📖 Documentation

| Documentation   |                                                                |
| --------------- | -------------------------------------------------------------- |
| [Contribute]    | How to contribute to the sadedeGel project and code base.          |

[contribute]: https://github.com/GlobalMaksimum/sadedegel/blob/master/CONTRIBUTING.md

## 💬 Where to ask questions

The SadedeGel project is maintained by [@globalmaksimum](https://github.com/GlobalMaksimum) AI team members
[@dafajon](https://github.com/dafajon),
[@askarbozcan](https://github.com/askarbozcan),
[@mccakir](https://github.com/mccakir) and 
[@husnusensoy](https://github.com/husnusensoy). 

| Type                     | Platforms                                              |
| ------------------------ | ------------------------------------------------------ |
| 🚨 **Bug Reports**       | [GitHub Issue Tracker]                                 |
| 🎁 **Feature Requests**  | [GitHub Issue Tracker]                                 |

[github issue tracker]: https://github.com/GlobalMaksimum/sadedegel/issues

## Features

* Several news datasets
  * Basic corpus
      * Raw corpus (`sadedegel.dataset.load_raw_corpus`)
      * Sentences tokenized corpus (`sadedegel.dataset.load_sentences_corpus`)  
      * Human annotated summary corpus (`sadedegel.dataset.load_annotated_corpus`)   
  * [Extended corpus](sadedegel/dataset/README.md)
      * Raw corpus (`sadedegel.dataset.extended.load_extended_raw_corpus`)
      * Sentences tokenized corpus (`sadedegel.dataset.extended.load_extended_sents_corpus`)
* ML based sentence boundary detector (**SBD**) trained for Turkish language (`sadedegel.dataset`)
* Various baseline summarizers
  * Position Summarizer
    * First Important Summarizer
    * Last Important Summarizer
  * Length Summarizer
  * Band Summarizer
  * Random Summarizer
  
* Various unsupervised/supervised summarizers
  * ROUGE1 Summarizer
  * Cluster Summarizer
  * Supervised Summarizer
 

📖 **For more details, refere to [sadedegel.ai](http://sadedegel.ai)**

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

When using pip it is generally recommended to install packages in a virtual
environment to avoid modifying system state:

```bash
python -m venv .env
source .env/bin/activate
pip install sadedegel
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

To use our ML based sentence boundary detector

```python
from sadedegel import Doc

doc = ("Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı. Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok "
"daha büyük ölçekte. Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu. IBM 650 Model I adını taşıyan bilgisayarın "
"satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı. Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı. "
"Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.")

Doc(doc).sents
```
```python
['Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı.',
 'Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok daha büyük ölçekte.',
 'Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu.',
 'IBM 650 Model I adını taşıyan bilgisayarın satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı.',
 'Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı.',
 'Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.']
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

## References
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
  * [Code Inspector](https://code-inspector.com) for allowing us to keep track of our code quality and technical debt.
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
