<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/dist/img/logo-2.png?s=280&v=4" width="125" height="125" align="right" /></a>

# Contribute to sadedeGel

Thanks for your interest in contributing to sadedeGel 🎉 The project is maintained
by maintained by [@globalmaksmum](https://github.com/GlobalMaksimum) AI team members
[@dafajon](https://github.com/dafajon),
[@askarbozcan](https://github.com/askarbozcan),
[@mccakir](https://github.com/mccakir) and 
[@husnusensoy](https://github.com/husnusensoy),
and we'll do our best to help you get started. This page will give you a quick
overview of how things are organised and most importantly, how to get involved.

## Table of contents

1. [Issues and bug reports](#issues-and-bug-reports)
2. [Contributing to the code base](#contributing-to-the-code-base)
3. [Code conventions](#code-conventions)
4. [Adding tests](#adding-tests)
5. [Dataset](#dataset)
6. [Hotfix Checklist](#hotfix-checklist)

## Issues and bug reports

First, [do a quick search](https://github.com/issues?q=is%3Aissue+user%3Aglobalmaksimum)
to see if the issue has already been reported. If so, it's often better to just
leave a comment on an existing issue, rather than creating a new one. Old issues
also often include helpful tips and solutions to common problems.

### Submitting issues

When opening an issue, use a **descriptive title** and include your
**environment** (operating system, Python version). If you've discovered a bug, you
can also submit a [regression test](#fixing-bugs) straight away. When you're
opening an issue to report the bug, simply refer to your pull request in the
issue body. A few more tips:

-   **Describing your issue:** Try to provide as many details as possible. What
    exactly goes wrong? _How_ is it failing? Is there an error?
    "XY doesn't work" usually isn't that helpful for tracking down problems. Always
    remember to include the code you ran and if possible, extract only the relevant
    parts and don't just dump your entire script. This will make it easier for us to
    reproduce the error.

-   **Sharing long blocks of code or logs:** If you need to include long code,
    logs or tracebacks, you can wrap them in `<details>` and `</details>`. This
    [collapses the content](https://developer.mozilla.org/en/docs/Web/HTML/Element/details)
    so it only becomes visible on click, making the issue easier to read and follow.

## Contributing to the code base

You don't have to be an NLP expert or Python pro to contribute, and we're happy
to help you get started. If you're new to sadedeGel, a good place to start is to read our 
test cases covering more than 80% what sadedeGel can do. 

### Getting started

To make changes to sadedeGel's code base, you need to fork then clone the GitHub repository
and build sadedeGel from source. You'll need to make sure that you have a
development environment consisting of a Python distribution including header
files, a compiler, [pip](https://pip.pypa.io/en/latest/installing/),
[virtualenv](https://virtualenv.pypa.io/en/stable/) and
[git](https://git-scm.com) installed. The compiler is usually the trickiest part.

```
python -m pip install -U pip
git clone https://github.com/GlobalMaksimum/sadedegel
cd sadedegel

python -m venv .env
source .env/bin/activate
export PYTHONPATH=`pwd`
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### Fixing bugs

When fixing a bug, first create an
[issue](https://github.com/GlobalMaksimum/sadedegel/issues) if one does not already exist.
The description text can be very short – we don't want to make this too
bureaucratic.

Next, create a test file named `test_issue[ISSUE NUMBER].py` in the
[`sadedegel/tests`](sadedegel/tests) folder. Test for the bug
you're fixing, and make sure the test fails. Next, add and commit your test file
referencing the issue number in the commit message. Finally, fix the bug, make
sure your test passes and reference the issue in your commit message.


## Code conventions

Code should loosely follow [pep8](https://www.python.org/dev/peps/pep-0008/).
sadedeGel uses [`flake8`](http://flake8.pycqa.org/en/latest/) for linting its
Python modules. If you've built sadedeGel from source, you'll already have both
tools installed.

### Code formatting

As of today sadedeGel still haven't standardized code formatting 😟
But it is highly probable that [`black`](https://github.com/ambv/black) 
will be our choice because of its tight integration with various 

### Code linting

[`flake8`](http://flake8.pycqa.org/en/latest/) is a tool for enforcing code
style. It scans one or more files and outputs errors and warnings. This feedback
can help you stick to general standards and conventions, and can be very useful
for spotting potential mistakes and inconsistencies in your code. The most
important things to watch out for are syntax errors and undefined names, but you
also want to keep an eye on unused declared variables or repeated
(i.e. overwritten) dictionary keys.

The [`.flake8`](.flake8) config defines the configuration we use for this
codebase. For example, we're not super strict about the line length, and we're
excluding very large files like lemmatization and tokenizer exception tables.

Ideally, running the following command from within the repo directory should
not return any errors or warnings:

```bash
make lint
```

#### Disabling linting

Sometimes, you explicitly want to write code that's not compatible with our
rules. For example, a module's `__init__.py` might import a function so other
modules can import it from there, but `flake8` will complain about an unused
import. And although it's generally discouraged, there might be cases where it
makes sense to use a bare `except`.

To ignore a given line, you can add a comment like `# noqa: F401`, specifying
the code of the error or warning we want to ignore. It's also possible to
ignore several comma-separated codes at once, e.g. `# noqa: E731,E123`. Here
are some examples:

```python
# The imported class isn't used in this file, but imported here, so it can be
# imported *from* here by another module.
from .submodule import SomeClass  # noqa: F401

try:
    do_something()
except:  # noqa: E722
    # This bare except is justified, for some specific reason
    do_something_else()
```

### Resources to get you started

-   [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) (python.org)
-   [CONTRIBUTING document of SpaCy](https://github.com/explosion/spaCy/blob/master/CONTRIBUTING.md) (explosion.ai)

## Adding tests

sadedeGel uses the [pytest](http://doc.pytest.org/) framework for testing. For more
info on this, see the [pytest documentation](http://docs.pytest.org/en/latest/contents.html).
All tests for sadedeGel modules and classes live in a flat directory
[`sadedeGel/tests`](sadedeGel/tests). We may choose to split this into subdirectories as the project grows.
To be interpreted and run, all test files and test functions need to be prefixed with `test_`.

When adding tests, make sure to use descriptive names, keep the code short and
concise and only test for one behaviour at a time. Try to `parametrize` test
cases wherever possible

Extensive tests that take a long time should be marked with `@pytest.mark.slow`.
Tests that require the model to be loaded should be marked with
`@pytest.mark.models`. Loading the models is expensive and not necessary if
you're not actually testing the model performance.

## Dataset

sadedeGel uses a few internal datasets to guarantee minimal data support. 
When contributing to our existing 100 document dataset (as of today), please ensure not to hurt dataset sanity.

For a minimal control for sanity check use

```bash
python -m sadedel.dataset -m validate
```

## Hotfix Checklist

- [ ] Start by creating an issue and ensure that what you are fixing is documented very clearly
- [ ] Start by creeating your hotfix branch usign `git flow hotfix start 0.<minor-release>.<next-hotfix-release>`. Ensure that your hotfix name confirms [SemVer](https://semver.org/) rules.
- [ ] Make your changes accordingly and commit. Note to use `[resolves #issue-number]` to let Github close your issue automatically.
- [ ] Ensure that you also update `__version__` variable in `sadedegel/about.py` with your hotfix release.
- [ ] Run unit tests
- [ ] Run linter, bandit and flake checks.
- [ ] Once all done push your branch to github. Ensure that remove naming confirms `hotfix/0.<minor-release>.<next-hotfix-release>` for commit masters' future investigation.
