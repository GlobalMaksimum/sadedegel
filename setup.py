from setuptools import setup
from os.path import dirname
import sys


def version():
    sys.path.insert(0, dirname(__file__))

    from sadedegel.about import __version__

    return __version__


with open('prod.requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sadedegel',
    # Needed to actually package something
    packages=['sadedegel', 'sadedegel.dataset', 'sadedegel.summarize', 'sadedegel.tokenize', 'sadedegel.tokenize.ml'],
    package_data={
        'sadedegel.dataset': ['raw/*.txt', 'sents/*.json'],
        'sadedegel.tokenize.ml': ['model/sbd.pickle']
    },
    # Needed for dependencies
    install_requires=install_requires,
    # *strongly* suggested for sharing
    version=version(),
    python_requires='>=3.5',
    entry_points='''
        [console_scripts]
        sadedegel=sadedegel.__main__:cli
    '''
)
