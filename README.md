<a href="https://explosion.ai"><img src="https://avatars0.githubusercontent.com/u/2204565?s=280&v=4" width="125" height="125" align="right" /></a>

# SadedeGel: An extraction based Turkish news summarizer

SadedeGel is a library for extraction-based news summarizer using pretrained BERT model.
Development of the library takes place as a part of [Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/)

💫 **Version 0.1 out now!**
[Check out the release notes here.](https://github.com/GlobalMaksimum/sadedegel/releases)


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
raw = load_sentence_corpus()

summary = nlp(raw[0])
summary = nlp(tokenized[0], sentence_tokenizer=False)
```

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
* Special thanks to [spaCy](https://github.com/explosion/spaCy) project for their work in showing us the way to implement a proper python module rather than merely explaining it.
    * We have borrowed many document and style related stuff from their code base :smile:  