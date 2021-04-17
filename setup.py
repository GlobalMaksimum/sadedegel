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
    packages=['sadedegel', 'sadedegel.bblock',
              'sadedegel.dataset', 'sadedegel.dataset.extended', 'sadedegel.dataset.profanity',
              'sadedegel.dataset.tweet_sentiment', 'sadedegel.dataset.tscorpus',
              'sadedegel.dataset.categorized_product_sentiment', 'sadedegel.dataset.customer_review',
              'sadedegel.dataset.hotel_sentiment', 'sadedegel.dataset.movie_sentiment',
              'sadedegel.dataset.product_sentiment', 'sadedegel.dataset.telco_sentiment',
              'sadedegel.summarize',
              'sadedegel.summarize.evaluate',
              'sadedegel.summarize.util', 'sadedegel.extension',
              'sadedegel.bblock.cli',
              'sadedegel.tokenize', 'sadedegel.ml',
              'sadedegel.server', 'sadedegel.metrics', 'sadedegel.prebuilt'],
    package_data={
        'sadedegel.dataset': ['raw/*.txt', 'sents/*.json', 'annotated/*.json'],
        'sadedegel.ml': ['model/sbd.pickle'],
        'sadedegel.prebuilt': ['model/*.joblib'],
        'sadedegel.bblock': ['data/bert/vocabulary.hdf5', 'data/simple/vocabulary.hdf5', 'data/icu/vocabulary.hdf5',
                             'data/stop-words.txt'],
        'sadedegel': ['default.ini']
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
        sadedegel-dataset-tscorpus=sadedegel.dataset.tscorpus.__main__:cli
        sadedegel-bblock=sadedegel.bblock.cli.__main__:cli
        sadedegel-summarize=sadedegel.summarize.evaluate.__main__:cli
        sadedegel-sbd=sadedegel.tokenize.__main__:cli
        sadedegel-server=sadedegel.server.__main__:server
        sadedegel-build-vocabulary=sadedegel.bblock.cli.__main__:build_vocabulary
    ''',
    extras_require={
        'w2v': ['gensim'],
        'bert': ['torch==1.5.1', 'transformers==3.0.0']
    }
)
