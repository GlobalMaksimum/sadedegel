from setuptools import setup

with open('prod.requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sadedegel',
    # Needed to actually package something
    packages=['sadedegel', 'sadedegel.dataset', 'sadedegel.summarize', 'sadedegel.tokenize'],
    package_data={
        'sadedegel.dataset': ['raw/*.txt', 'sents/*.json']
    },
    # Needed for dependencies
    install_requires=install_requires,
    # *strongly* suggested for sharing
    version='0.2',
    python_requires='>=3.5',
    entry_points='''
        [console_scripts]
        sadedegel=sadedegel.__main__:cli
    '''
)
