<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/dist/img/logo-2.png?s=280&v=4" width="125" height="125" align="right" /></a>

# SadedeGel Datasets

SadedeGel provides 2 major datasets

## Basic Dataset
Basic dataset consists of ~100 documents and it is **installed** with the package and ready to be used.

```python
from sadedegel.dataset import load_raw_corpus, load_sentence_corpus


raw = load_raw_corpus()
sents = load_sentence_corpus()
```

There are 3 parts of this corpus
1. **Raw Corpus** is made of `txt` files consisting raw news documents scraped by [Sadedegel Scraper](https://github.com/GlobalMaksimum/sadedegel-scraper)
2. **Sentences Corpus** is made of `json` files consisting sentence boundary detected documents in raw corpus. 
Boundary detection is initialized by a automatic (ML/rule based) sentence tokenizer but corrected by human annotators  

    * Refer to [Sentences Corpus Tokenize](#sentence-corpus-tokenize)
    * Refer to [Sentences Corpus Validate](#sentence-corpus-validate)

3. **Annotated Corpus** is made of `json` files. TODO

### Sentence Corpus Tokenize

Preprocessing stage of sentence tokenization before human annotator.

```bash
python -m sadedegel.dataset tokenize
```

### Sentence Corpus Validate

Validation process ensures that hand annotated sentence tokenization does not violate any of the following span rule

1. All sentences should cover a span in corresponding raw document.
2. All sentences spans should be stored in order at `sentences` (of list type) member of `json` document.

```bash
python -m sadedegel.dataset validate
```
    


## Extended Dataset  
### Download Dataset 

You can download extended dataset using 

```bash
python -m sadedegel.dataset.extended download
```

Sub command requires two flags to access GCS buckets 
* `access-key`
* `secret-key`

Those can be passed in 3 ways:
1. Set `sadedegel_access_key` and/or `sadedegel_secret_key` in you environment.
2. Pass `--access-key` and `--secret-key` options in commandline
3. You will be prompted to provide them if they are not provided at 1 or 2.


### Check Metadata

You can assert your extended dataset using 

```bash
python -m sadedegel.dataset.extended metadata 
```

If everything is OK you will get

```bash
{'bytes': {'raw': 170014810, 'sents': 210067269},
 'count': {'raw': 36131, 'sents': 36131}}
```

If there is a problem with base directory you will get a similar warning

```
~/.sadedegel_data not found.

Ensure that you have properly downloaded extended corpus using
         
            python -m sadedegel.dataset.extended download --access-key xxx --secret-key xxxx
            
        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.
```



