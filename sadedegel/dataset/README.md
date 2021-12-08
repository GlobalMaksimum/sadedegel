<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/assets/img/logo-2.png?s=280&v=4" width="125" height="125" align="right" /></a>

# SadedeGel Datasets

SadedeGel provides various datasets to consolidate various NLP data sources for Turkish Language,
train prebuilt models, and benchmark models.

## `raw` Sadedegel News Corpus

[raw](raw/) sadedegel news corpus is a relatively small news corpus (roughly 100 documents) gathered using  [scraper]
and **installed** with with sadedegel installation. No extra download required.


### Using Corpus

```python
from sadedegel.dataset import load_raw_corpus

raw = load_raw_corpus()
```

## `sents` Sadedegel News Corpus

[sents](sents/) sadedegel news corpus is the sentences boundary detected (human annotation) version of [raw](raw/) corpus
and also **installed** with with sadedegel installation. No extra download required.

### Using Corpus

```python
from sadedegel.dataset import load_sentence_corpus

sents = load_sentence_corpus()
```

## `annotated` Sadedegel News Corpus

[annotated](annotated/) sadedegel news corpus is the sentences importance (aka `relevance`) annotated (human annotation) version of [sentences](sentences/) corpus
and also **installed** with with sadedegel installation. No extra download required.

### Using Corpus

```python
from sadedegel.dataset import load_annotated_corpus

sents = load_annotated_corpus()
```

## `extended` Sadedegel News Corpus

[extended](extended/) is a collection of corpus (corpura) that can be defined as the extended version of [raw](raw/) and [sents](sents/):

* [extended](extended/) **raw** is simply a larger collection of news documents collected by [scraper]
* [extended](extended/) **sents** is generated using [extended](extended/) **raw** and ML based sentence boundary detector trained over [sents](sents/) corpus


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




* Refer to [Sentences Corpus Tokenize](#sentence-corpus-tokenize)



### Sentence Corpus Tokenize

Preprocessing stage of sentence tokenization before human annotator.

```bash
python -m sadedegel.dataset tokenize
```

### Sentence Corpus Validate

Validation process ensures that hand annotated sentence tokenization does not violate any of the following span rule

1. All sentences should cover a span in corresponding raw document.
2. All sentences spans should be stored in order at `sentences` (of list type) field of `json` document.

```bash
python -m sadedegel.dataset validate
```

## `tscorpus`

[tscorpus] is an invaluable contribution by [Taner Sezer].

Corpora consist of two corpus
* [tscorpus] raw
* [tscorpus] tokenized, word tokenized version of [tscorpus] raw corpus

 Each corpus is splited into subsections per news category:
* art_culture
* education
* horoscope
* life_food
* politics
* technology
* economics
* health
* life
* magazine
* sports
* travel

[tscorpus] allows us to
1. Verify/Calibrate word tokenizers (bert, simple, etc.) available in sadedegel
2. Ship a prebuilt news classifier.

### Using Corpora

To load [tscorpus] for word tokenization tasks

```python
from sadedegel.dataset.tscorpus import load_tokenization_tokenized, load_tokenization_raw

raw = load_tokenization_raw()
tok = load_tokenization_tokenized()
```

Refer [tokenizer](../bblock/TOKENIZER.md) for details.

To load [tscorpus] for classification tasks

```python
from sadedegel.dataset.tscorpus import load_classification_raw

data = load_classification_raw()
```

Refer  [news classification](../prebuilt/README.md) for details

## `profanity`

Corpus used in [SemEval-2020 Task 12](https://arxiv.org/pdf/2006.07235.pdf) to implement profanity classifier over
Turkish tweeter dataset.

Training dataset contains 31277 documents, whereas test dataset consists of 3515 documents.

### Using Corpus

```python
from sadedegel.dataset.profanity import load_offenseval_train, load_offenseval_test_label, load_offenseval_test

tr = load_offenseval_train()
tst = load_offenseval_test()
tst_label = load_offenseval_test_label()

next(tr)

# {'id': 20948,
# 'tweet': "@USER en güzel uyuyan insan ödülü jeon jungkook'a gidiyor...",
# 'profanity_class': 0}

next(tst)
# {'id': 41993, 'tweet': '@USER Sayın başkanım bu şekilde devam inşallah👏'}

next(tst_label)
# {'id': 41993, 'profanity_class': 0}
```

For more details please refer [tweet profanity](../prebuilt/README.md)

## `tweet_sentiment`

[Twitter Dataset](https://www.kaggle.com/mrtbeyz/trke-sosyal-medya-paylam-veri-seti) is another corpus used to build prebuilt
tweeter sentiment classifier.

For more details please refer [tweet sentiment](../prebuilt/README.md)

## `customer_review`

Customer review classification corpus consists of 323479 training and 107827 test instances which contains customer reviews in 
the `text` field and shopping category that the review refers to in the `review_class` field. 

There are 32 unique class labels for this corpus which are mapped to their respective IDs on `CLASS_VALUES` dict. 

### Using Corpus
**Usage**

```python
from sadedegel.dataset.customer_review import load_train
from sadedegel.dataset.customer_review import load_test
from sadedegel.dataset.customer_review import load_test_label

next(load_train())

# Out[6]: 
# {'id': 'cb60a760-cfeb-44e8-abb1-4cbcd6814c64',
#  'text': 'Hipp 1 Mama Bebeğime Zarar Verdi,"Hipp 1devam sütü bebeğimde inanılmaz derecede gaz ve kusmaya neden oldu! Kızımda yıllar önce yine Hipp in devam sütü, pirinç maması, ek gıdaları gofret ve bisküvileri, yine aynı şekilde erik meyve püreleri her şeyini kullanıyordum. Hiçbir şekilde böyle bir sorunla karşılaşmamıştım. Ancak bu sefer al...Devamını oku"',
#  'review_class': 1}

next(load_test())

# {'id': '97fdc0de-98e1-4577-9d7f-86cb71a49bbe',
#  'text': 'Samsung Garanti Garanti Değil!,BDH ile anlaşılmış garanti şirketi olarak ama hiçbir şekilde ne onlar kabul ediyor hatalarını ne de Samsung üstleniyor. Ben bıktım servise kendimi iletemediğimi sanıyordum içerisinde kağıt ile şikayetlerimi gönderdim ama maalesef okumayıp bir kez daha beni salak yerine koyup bu sefer eğik olan kasam...Devamını oku'}

next(load_test_label())

# {'id': '97fdc0de-98e1-4577-9d7f-86cb71a49bbe', 'review_class': 4}
```

```python
from sadedegel.dataset.customer_review import CLASS_VALUES

CLASS_VALUES[1]

# Out[2]: 'alisveris'
```
## `telco_sentiment`

Telecom Sentiment dataset is an open sourced tweet sentiment corpus that includes tweets referring to a certain telecom
company. It is a social media commentary dataset used to evaluate sentiments over a certain brand.
Dataset [source](http://www.kemik.yildiz.edu.tr/veri_kumelerimiz.html)
and [paper](https://ieeexplore.ieee.org/document/8554037) are provided.

### Using Corpus

```python
from sadedegel.dataset.telco_sentiment import load_telco_sentiment_train
from sadedegel.dataset.telco_sentiment import load_telco_sentiment_test
from sadedegel.dataset.telco_sentiment import load_telco_sentiment_test_label

import pandas as pd

train_raw = load_telco_sentiment_train()
test_raw = load_telco_sentiment_test()
target_raw = load_telco_sentiment_test_label()

train_df = pd.DataFrame().from_records(train_raw)
test_df = pd.DataFrame().from_records(test_raw)
target_df = pd.DataFrame().from_records(target_raw)
```

## `movie_sentiment`

[Movie sentiment dataset](https://www.kaggle.com/mustfkeskin/turkish-movie-sentiment-analysis-dataset) is a corpus of **
entertainment** domain.

It contains 42975 instances of movie reviews with `POSITIVE` and `NEGATIVE` sentiments as a class label.

### Using Corpus

```python
from sadedegel.dataset.movie_sentiment import load_movie_sentiment_train
from sadedegel.dataset.movie_sentiment import load_movie_sentiment_test
from sadedegel.dataset.movie_sentiment import load_movie_sentiment_test_label

train = load_movie_sentiment_train()
test = load_movie_sentiment_test()
test_label = load_movie_sentiment_test_label()
```

## `hotel_sentiment`

Hotel sentiment data is part of [HUMIR dataset](http://humirapps.cs.hacettepe.edu.tr/tsad.aspx), which is a combination
of hotel and movie reviews. This implementation contains reviews of type 'Hotel Review'.

It contains 11,600 instances with `POSITIVE` and `NEGATIVE` sentiments as a class label. The train and
test split is based on the split present in HUMIR dataset.

### Using Corpus

```python
from sadedegel.dataset.hotel_sentiment import load_hotel_sentiment_train
from sadedegel.dataset.hotel_sentiment import load_hotel_sentiment_test
from sadedegel.dataset.hotel_sentiment import load_hotel_sentiment_test_label

train = load_hotel_sentiment_train()
test = load_hotel_sentiment_test()
test_label = load_hotel_sentiment_test_label()
```

## `categorized_product_sentiment`

This corpus contains 5600 instances of customer product reviews from E-commerce sites. Reviews contain two sets of class labels. First label is `sentiment_class` which contains `[POSITIVE, NEGATIVE]` sentiment of the review. Second label is `product_category` which contains `["Kitchen", "DVD", "Books", "Electronics"]` as the category of the product being reviewed. Each product category contains 1400 instances. The dataset is material to the research [paper](https://sentic.net/wisdom2013pechenizkiy.pdf) by Demirtaş and Pechenizkiy.

Number of instances in each `product_category` grouped by `sentiment_class`:

|             | `Kitchen` | `Books` | `DVD`    | `Electronics` |
| :---        | :----:  | :---: |  :---: | :---:  |
| **`POSITIVE`**    | 700     | 700   |  700   |  700   |
| **`NEGATIVE`**    | 700     | 700   |  700   |  700   |

```python
import pandas as pd

from sadedegel.dataset.categorized_product_sentiment import load_categorized_product_sentiment_train
from sadedegel.dataset.categorized_product_sentiment import SENTIMENT_CLASS_VALUES, PRODUCT_CATEGORIES

raw = load_categorized_product_sentiment_train()

next(raw)

# Out [0]: {'id': 'bac3a153-397e-4c90-aaec-c9dfa51a9784', 'text': 'ürünün tedarik edilme süreci biraz uzasa da beklediğime değdi, set cam ağırlıklı olmasına rağmen sağlam olarak elime ulaştı. almayı düşünenlere tavsiye ederim, set beklentilerinizi karşılıyor...', 'product_category': 0, 'sentiment_class': 0}


df = pd.DataFrame().from_records(raw)

# Load Subsets

dvd = load_categorized_product_sentiment_train('DVD')
kitchen = load_categorized_product_sentiment_train('Kitchen')
books = load_categorized_product_sentiment_train('Books')
electronics = load_categorized_product_sentiment_train('Electronics')

# Mappings

SENTIMENT_CLASS_VALUES[0]
# Out [0]: 'POSITIVE'

PRODUCT_CATEGORIES[0]
# Out [0]: 'Kitchen'
```

## `food_reviews`

This dataset contains over 500k food service reviews. It is combination of these two datasets can be found 
[here](https://www.kaggle.com/berkaycokuner/yemek-sepeti-comments) and
[here](https://www.kaggle.com/dgknrsln/yorumsepeti).

The dataset instances contain the reviews for food services and ratings made by users. Review points are discretized into `POSITIVE` and `NEGATIVE` classes by taking minimum of three
scoring fields (Speed, Service and Flavor) and if it's lower than predefined threshold
(which is 7) then it gets labelled as `NEGATIVE` else `POSITIVE`.

Another way to use this dataset is the "Multilabel version". Instead of taking minimum out of three
you can use scores by field or all together.

For both methods sadedegel has pretrained models under `sadedegel.prebuilt`.

### Using Corpus
#### To Get the Data
```python
from sadedegel.dataset.food_review import load_food_review_train
from sadedegel.dataset.food_review import load_food_review_test

train = load_food_review_train()
test = load_food_review_test()
```


[scraper]: https://github.com/GlobalMaksimum/sadedegel-scraper

[Taner Sezer]: https://github.com/tanerim

[tscorpus]: tscorpus/
