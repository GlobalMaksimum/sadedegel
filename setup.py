from setuptools import setup

with open('prod.requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='sadedegel',
    url='https://github.com/GlobalMaksimum/sadedegel',
    author='Global Maksimum',
    author_email='info@globalmaksimum.com',
    # Needed to actually package something
    packages=['sadedegel', 'sadedegel.dataset', 'sadedegel.summarize', 'sadedegel.tokenize'],
    package_data={
        'sadedegel.dataset': ['raw/*.txt', 'sents/*.json']
    },
    # Needed for dependencies
    install_requires=install_requires,
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Extraction-based Turkish news summarizer.',
    python_requires='>=3.5',
    entry_points='''
        [console_scripts]
        sadedegel=sadedegel.__main__:cli
    '''
)
