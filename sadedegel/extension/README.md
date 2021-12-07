### `scikit-learn` API

`sadedegel` contains `sklearn.pipeline.Pipeline` compatible feature transformers. Users can build serializable pipeline objects that make use of sadedegel feature extractors including:

- `Text2Doc`
- `HashVectorizer` 
- `TfidfVectorizer`
- `BM25Vectorizer`
- `PreTrainedVectorizer`

---
`Text2Doc` converts raw string to a `sadedegel.bblock.Doc` object.  


`TfidfVectorizer` and `BM25Vectorizer` resorts to a `sadedegel.bblock.Vocabulary` object that loads a built-in or user-built vocabulary dump. For building custom a vocabulary dump, revisit [link here]().

```python
from sadedegel.dataset.tweet_sentiment import load_tweet_sentiment_train 
from sadedegel.extension.sklearn import Text2Doc, TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import pandas as pd

import joblib


data = pd.DataFrame().from_records(load_tweet_sentiment_train())
X, y = data["tweet"], data["sentiment_class"]

pipeline = Pipeline([("text2doc", Text2Doc(tokenizer="icu", emoticon=True, emoji=True, hashtag=True, mention=True)),
                     ("tfidf", TfidfVectorizer()),
                     ("model", LogisticRegression(C=0.123, max_iter=5000))])        
pipeline.fit(X, y)

joblib.dump(pipeline, "sg_tweet_sentiment_pipeline.joblib")      
```

---
 `HashVectorizer` does not require any vocabulary lookup as it hashes raw tokens to a sparse vector on the fly. For a corpus with a large vocabulary, collisions may affect the performance. 
 
 Normalization of tokens might help with collision issue. Currently `sadedegel` library does not have a built-in normalizer, however agglunative structure of Turkish is used as an advantage. 
 `sadedegel` assumes that pruning a pre-determined length of prefix from tokens will help normalizing many Turkish words to their respective lemmas. 
 
 Powered by this assumption, using `prefix_range` argument, the `HashVectorizer` hashes the tokens after their prefixes of determined lengths are removed.

```python
from sadedegel.extension.sklearn import Text2Doc, HashVectorizer

pipeline = Pipeline([("text2doc", Text2Doc(tokenizer="icu", emoticon=True, emoji=True, hashtag=True, mention=True)),
                     ("tfidf", HashVectorizer(prefix_range=(3, 5))),
                     ("model", LogisticRegression(C=0.123, max_iter=5000))])        
```

---
#### `PreTrainedVectorizer` for HuggingFace Hub models

With the help of `sentence-transformers` dependency, `sadedegel.bblock.Doc` object has the ability to encode sentences and documents into dense embeddings. These are produced by a single forward pass throuhg pre-trained transformer based architecture. Currently `sadedegel` supports `bert_32k_cased`, `bert_128k_cased`, `bert_32k_uncased`, `bert_128k_uncased`, `distilbert` 