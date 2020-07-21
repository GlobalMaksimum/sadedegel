<a href="http://sadedegel.ai"><img src="https://avatars0.githubusercontent.com/u/2204565?s=280&v=4" width="125" height="125" align="right" /></a>

# SadedeGel: An extraction based Turkish news summarizer

SadedeGel is a library for extraction-based news summarizer using pretrained BERT model.
Development of the library takes place as a part of [Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/)

💫 **Version 0.3.1 out now!**
[Check out the release notes here.](https://github.com/GlobalMaksimum/sadedegel/releases)


![Python package](https://github.com/GlobalMaksimum/sadedegel/workflows/Python%20package/badge.svg)
[![pypi Version](https://img.shields.io/pypi/v/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![PyPi downloads](https://img.shields.io/pypi/dm/sadedegel?style=plastic&logo=PyPI)](https://pypi.org/project/sadedegel/)
[![Coverage](https://img.shields.io/coveralls/github/globalmaksimum/sadedegel?style=plastic)]()
[![Dependencies](https://img.shields.io/librariesio/github/globalmaksimum/sadedegel?style=plastic&logo=Python)]()
[![License](https://img.shields.io/pypi/l/sadedegel)]()
[![Commit Month](https://img.shields.io/github/commit-activity/m/globalmaksimum/sadedegel?style=plastic&logo=GitHub)]()
[![Commit Week](https://img.shields.io/github/commit-activity/w/globalmaksimum/sadedegel?style=plastic&logo=GitHub)]()
[![Last Commit](https://img.shields.io/github/last-commit/globalmaksimum/sadedegel?style=plastic&logo=GitHub)]()


## 📖 Documentation

| Documentation   |                                                                |
| --------------- | -------------------------------------------------------------- |
| [Contribute]    | How to contribute to the sadedeGel project and code base.          |

[contribute]: https://github.com/GlobalMaksimum/sadedegel/blob/master/CONTRIBUTING.md

## 💬 Where to ask questions

The SadedeGel project is maintained by [@globalmaksmum](https://github.com/GlobalMaksimum) AI team members
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

Coming soon...

📖 **For more details, see the

Coming soon...

## Install sadedeGel

- **Operating system**: macOS / OS X · Linux · Windows (Cygwin, MinGW, Visual
  Studio)
- **Python version**: 3.5+ (only 64 bit)
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

### conda

Coming soon...


### Quickstart with SadedeGel

To load SadedeGel, use `sadedegel.load()`

```python
import sadedegel
from sadedegel.dataset import load_sentence_corpus, load_raw_corpus

nlp = sadedegel.load()
tokenized = load_sentence_corpus()
raw = load_raw_corpus()

summary = nlp(raw[0])
summary = nlp(tokenized[0], sentence_tokenizer=False)
```

To use our ML based sentence boundary detector

```python
from sadedegel.tokenize import Doc

doc = ("Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı. Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çok"
"daha büyük ölçekte. Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu. IBM 650 Model I adını taşıyan bilgisayarın"
"satın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı. Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı. "
"Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.")

Doc(doc).sents
```
```python
['Bilişim sektörü, günlük devrimlerin yaşandığı ve hızına yetişilemeyen dev bir alan haline geleli uzun bir zaman olmadı.',
 'Günümüz bilgisayarlarının tarihi, yarım asırı yeni tamamlarken; yaşanan gelişmeler çokdaha büyük ölçekte.',
 'Türkiye de bu gelişmelere 1960 yılında Karayolları Umum Müdürlüğü (şimdiki Karayolları Genel Müdürlüğü) için IBM’den satın aldığı ilk bilgisayarıyla dahil oldu.',
 'IBM 650 Model I adını taşıyan bilgisayarınsatın alınma amacı ise yol yapımında gereken hesaplamaların daha hızlı yapılmasıydı.',
 'Türkiye’nin ilk bilgisayar destekli karayolu olan 63 km uzunluğundaki Polatlı - Sivrihisar yolu için yapılan hesaplamalar IBM 650 ile 1 saatte yapıldı.',
 'Daha öncesinde 3 - 4 ayı bulan hesaplamaların 1 saate inmesi; teknolojinin, ekonomik ve toplumsal dönüşüme büyük etkide bulunacağının habercisiydi.']
```

#### SadedeGel Server
In order to integrate with your applications we provide a quick summarizer server with sadedeGel.

```bash
python3 -m sadedegel.server 
```

Refer to self documenting APIs for details (http://localhost:8000/docs or http://localhost:8000/redoc by default)

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
### Software Engineering
* Special thanks to [spaCy](https://github.com/explosion/spaCy) project for their work in showing us the way to implement a proper python module rather than merely explaining it.
    * We have borrowed many document and style related stuff from their code base :smile:
    
### Machine Learning (ML), Deep Learning (DL) and Natural Language Processing (NLP)
* Resources on Extractive Text Summarization:

    * [Leveraging BERT for Extractive Text Summarization on Lectures](https://arxiv.org/abs/1906.04165)  by Derek Miller
    * [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf) by Yang Liu

* Other NLP related references

    * [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013.pdf)
