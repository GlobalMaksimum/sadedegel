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
    packages=['sadedegel', 'sadedegel.bblock', 'sadedegel.dataset', 'sadedegel.dataset.extended', 'sadedegel.summarize',
              'sadedegel.tokenize', 'sadedegel.ml',
              'sadedegel.server', 'sadedegel.metrics'],
    package_data={
        'sadedegel.dataset': ['raw/*.txt', 'sents/*.json', 'annotated/*.json'],
        'sadedegel.ml': ['model/sbd.pickle']
    },
    # Needed for dependencies
    install_requires=install_requires,
    # *strongly* suggested for sharings
    version=version(),
    python_requires='>=3.6',
    entry_points='''
        [console_scripts]
        sadedegel=sadedegel.__main__:cli
        sadedegel-dataset=sadedegel.dataset.__main__:cli
        sadedegel-dataset-extended=sadedegel.dataset.extended.__main__:cli
        sadedegel-summarize=sadedegel.summarize.__main__:cli
        sadedegel-sbd=sadedegel.tokenize.__main__:cli
        sadedegel-server=sadedegel.server.__main__:server
    '''
)
