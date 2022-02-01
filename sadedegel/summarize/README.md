## SadedeGel Model

We model a human annotator for summarization task as follows:
1. A human annotator reads a news document and 
choose to drop out some sentences in it (any number in any round) based on his/her prior knowledge of language and news' domain.
2. Annotator reads left over document again and repeat 1. until no sentences are left in the news document.

We keep track of human annotator behaviour with [SadedeGel Annotator](https://github.com/GlobalMaksimum/sadedegel-annotator) 
by recoding the **Round** of each sentences in which it is eliminated.

Later a sentence is eliminated, higher its relative score is within a given news document. 

## Summarizer Usage

SadedeGel summarizers share same interface. 

First a `sadedegel.summarize.ExtractiveSummarizer` instance is constructed. 
```python
from sadedegel.summarize import LengthSummarizer, TFIDFSummarizer, DecomposedKMeansSummarizer

lsum = LengthSummarizer(normalize=True)
tfidf_sum = TFIDFSummarizer(normalize=True)
kmsum = DecomposedKMeansSummarizer(n_components=200, n_clusters=10)
```

Create a `sadedegel.Document` instance from the single document to be summarized.
```python
from sadedegel import Doc

d = Doc("ABD'li yayın organı New York Times, yaklaşık 3 ay içinde kullanıcı sayısını sıfırdan milyonlara çıkaran kelime oyunu Wordle’ı satın aldığını duyurdu. New York Times kısa bir süre önce de spor haberleri sitesi The Athletic'i satın almak için 550 milyon doları gözden çıkarmış ve bu satın alma ile birlikte 1.2 milyon abone kazanmıştı. ...")
```

For obtaining a summary of k sentences where k < n_sentences. Call the instance with a `Document` object or `List[Sentences]`

```python
summary1 = lsum(d, k=2)
summary2 = tfidf_sum(d, k=4)
summary3 = kmsum(d, k=5)
```
Alternatively you can obtain the relevance score of all sentences that is used to rank them to before selecting top k sentences.

```python
relevance_scores = kmsum.predict(d)
```

#### Supervised Ranker
All sadedegel summarizers work either with unsupervised or rule based methods to rank sentences before extracting top k as the summary. In the new release we are providing a ranker model that is trained on **SadedeGel Annotated Corpus** that has documents where each sentence has relevance label assigned by human annotators through a process of repetitive elimination.

Ranker uses document-sentence embedding pairs from transformer based pre-trained models as features. Future releases will accomodate BoW based and decomposition based embeddings as well. 
For possible pre-trained embedding types supported by sadedegel are `bert_32k_cased`, `bert_128k_cased`, `bert_32k_uncased`, `bert_128k_uncased`, `distilbert`.

```python
from sadedegel.summarize import SupervisedSentenceRanker

ranker = SupervisedSentenceRanker(vector_type="bert_32k_cased")
```

Supervised Ranker can be tuned for optimal performance over an embedding type and summarization percentage. Current ranker is optimized with `bert_128k_based` for average summarization performance over 10%, 50% and 80% of full document length.

**Example**: Specific fine-tuning for short summaries with a smaller embedding extraction model.
```python
from sadedegel.summarize.supervised import RankerOptimizer

fine_tuner = RankerOptimizer(vector_type="distilbert",
                             summarization_perc=0.1,
                             n_trials=20)

fine_tuner.optimize()
``` 

## Summarizer Performance 

Given this [Model Definition](#sadedegel-model), 
we use a ranking metric ([Normalized Discounted Cumulative Gain]) 
to evaluate different summarizers over independent dataset(s).

[Normalized Discounted Cumulative Gain] is a very intuitive metric by its definition. 
It simply measures an algorithm's success based on the ratio of two things

* Algorithm's choice of best k sentences among M sentences (total `relevance` score obtained with this k sentence).
* Best k sentences among M sentences with respect to 
ground truth human annotation (Best possible total `relevance` score that can be obtained with k sentences).

[Normalized Discounted Cumulative Gain]: https://en.wikipedia.org/wiki/Discounted_cumulative_gain


### Performance Table

#### Release 0.21.1
| Method           | Parameter                                                                                                                                |   ndcg(optimized for k=0.1) |   ndcg(optimized for k=0.5) |   ndcg(optimized for k=0.8) |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| SupervisedSentenceRanker | `{"vector_type": "bert_128k_cased"}`                                                                                                                         |        0.7620 |        0.7269 |        0.8163 |

#### Release 0.18

By 0.18 we have significantly changed the way we evaluate our summarizers. 

That's because including the parameter combinations we have more than 10.000 models to be evaluated with grid search
Instead  we now use a RandomSampler (which might further be improved by using libraries like [optuna](https://optuna.org/)) and 
store only Top-5 for 10%, 50% and 80% document length summaries.

##### Top-5 by ndcg(k=0.1)
| Method           | Parameter                                                                                                                                |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| BM25Summarizer | `{"b": 0.5976703637605387, "delta": 0.7125956761539498, "drop_punct": false, "drop_stopwords": true, "drop_suffix": true, "idf_method": "smooth", "k1": 2.4953802410827244, "lowercase": true, "tf_method": "log_norm"}` |        0.6866 |        0.7631 |        0.8534 |
| LengthSummarizer | `{"mode": "char"}`                                                                                                                         |        0.6856 |        0.7584 |        0.8520 |
| TFIDFSummarizer  | `{"tf_method": "binary", "lowercase": false, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": false, "drop_punct": true}`    |        0.6853 |        0.7611 |        0.8529 |
| Rouge1Summarizer | `{"metric": "recall"}`                                                                                                                     |        0.6851 |        0.7585 |        0.8488 |
| TFIDFSummarizer  | `{"tf_method": "log_norm", "lowercase": false, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": false, "drop_punct": false}` |        0.6850 |        0.7626 |        0.8541 |

##### Top-5 by ndcg(k=0.5)
| Method          | Parameter                                                                                                                                                                                                          |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| TFIDFSummarizer | `{"tf_method": "raw", "lowercase": true, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": true, "drop_punct": false}`                                                                                  |        0.6816 |        0.7638 |        0.8550 |
| BM25Summarizer | `{"b": 0.7514847848610613, "delta": 1.0171413823294055, "drop_punct": false, "drop_stopwords": true, "drop_suffix": true, "idf_method": "smooth", "k1": 2.020765846071259, "lowercase": true, "tf_method": "double_norm"}` |        0.6794 |        0.7632 |        0.8531 |
| TFIDFSummarizer | `{"tf_method": "log_norm", "lowercase": false, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": false, "drop_punct": false}`                                                                           |        0.6850 |        0.7626 |        0.8541 |
| TFIDFSummarizer | `{"tf_method": "log_norm", "lowercase": true, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": false, "drop_punct": false}`                                                                            |        0.6850 |        0.7626 |        0.8541 |
| BM25Summarizer  | `{"b": 0.7247476077499047, "delta": 1.085392166316497, "drop_punct": true, "drop_stopwords": true, "drop_suffix": true, "idf_method": "smooth", "k1": 1.3491012873595416, "lowercase": true, "tf_method": "binary"}` |        0.6802 |        0.7622 |        0.8525 |

##### Top-5 by ndcg(k=0.8)
| Method          | Parameter                                                                                                                                   |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| TFIDFSummarizer | `{"tf_method": "raw", "lowercase": true, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": true, "drop_punct": false}`           |        0.6816 |        0.7638 |        0.8550 |
| TFIDFSummarizer | `{"tf_method": "raw", "lowercase": false, "idf_method": "probabilistic", "drop_suffix": true, "drop_stopwords": true, "drop_punct": true}`    |        0.6778 |        0.7615 |        0.8544 |
| TFIDFSummarizer | `{"tf_method": "binary", "lowercase": false, "idf_method": "probabilistic", "drop_suffix": true, "drop_stopwords": true, "drop_punct": true}` |        0.6776 |        0.7618 |        0.8542 |
| TFIDFSummarizer | `{"tf_method": "double_norm", "lowercase": true, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": true, "drop_punct": false}`   |        0.6823 |        0.7608 |        0.8542 |
| TFIDFSummarizer | `{"tf_method": "log_norm", "lowercase": false, "idf_method": "smooth", "drop_suffix": true, "drop_stopwords": false, "drop_punct": false}`    |        0.6850 |        0.7626 |        0.8541 |


#### Release 0.17.1

We have significantly changed the way we calculate `raw_tf` (in return affecting the way we calculate all other `tf`s based on that), 
completely messing up the summarizer scoreboard. (Check [issue #201](https://github.com/GlobalMaksimum/sadedegel/issues/201) for details)
 
This will be fixed once we refactor our `sadedegel-summarizer evaluate` flow, which is no longer simple enough with the increasing number of summarizers (high variation due to parametrization) 

| Method & Tokenizer                                                                       |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| Random Summarizer - simple                                                               |        0.5566 |        0.6516 |        0.7695 |
| FirstK Summarizer - simple                                                               |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - simple                                                                |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - simple                                                          |        0.6697 |        0.7498 |        0.8433 |
| Band(k=2) Summarizer - simple                                                            |        0.4752 |        0.6275 |        0.7445 |
| Band(k=3) Summarizer - simple                                                            |        0.4911 |        0.6285 |        0.7526 |
| Band(k=6) Summarizer - simple                                                            |        0.4944 |        0.6418 |        0.7601 |
| Rouge1 Summarizer (precision) - simple                                                   |        0.4924 |        0.6298 |        0.7647 |
| Rouge1 Summarizer (recall) - simple                                                      |        0.6726 |        0.7558 |        0.8482 |
| Length Summarizer (char) - simple                                                        |        0.6753 |        0.7577 |        0.8502 |
| Length Summarizer (token) - simple                                                       |        0.6805 |        0.7575 |        0.8510 |
| Random Summarizer - bert                                                                 |        0.5497 |        0.6587 |        0.7744 |
| FirstK Summarizer - bert                                                                 |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - bert                                                                  |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - bert                                                            |        0.6833 |        0.7574 |        0.8488 |
| Band(k=2) Summarizer - bert                                                              |        0.4752 |        0.6275 |        0.7445 |
| Band(k=3) Summarizer - bert                                                              |        0.4911 |        0.6285 |        0.7526 |
| Band(k=6) Summarizer - bert                                                              |        0.4944 |        0.6418 |        0.7601 |
| Rouge1 Summarizer (precision) - bert                                                     |        0.5295 |        0.6500 |        0.7748 |
| Rouge1 Summarizer (recall) - bert                                                        |        0.6851 |        0.7585 |        0.8488 |
| Length Summarizer (char) - bert                                                          |        0.6843 |        0.7588 |        0.8483 |
| Length Summarizer (token) - bert                                                         |    **0.6856** |        0.7584 |        **0.8520** |
| KMeans Summarizer - bert                                                                 |        0.6599 |        0.7434 |        0.8344 |
| AutoKMeans Summarizer - bert                                                             |        0.6608 |        0.7418 |        0.8333 |
| DecomposedKMeans Summarizer - bert                                                       |        0.6579 |        0.7440 |        0.8341 |
| TextRank(0.05) Summarizer (BERT) - bert                                                  |        0.6212 |        0.7010 |        0.8000 |
| TextRank(0.15) Summarizer (BERT) - bert                                                  |        0.6226 |        0.7004 |        0.7999 |
| TextRank(0.30) Summarizer (BERT) - bert                                                  |        0.6223 |        0.7001 |        0.7999 |
| TextRank(0.5) Summarizer (BERT) - bert                                                   |        0.6232 |        0.7005 |        0.7998 |
| TextRank(0.6) Summarizer (BERT) - bert                                                   |        0.6213 |        0.6993 |        0.7993 |
| TextRank(0.7) Summarizer (BERT) - bert                                                   |        0.6212 |        0.6991 |        0.7991 |
| TextRank(0.85) Summarizer (BERT) - bert                                                  |        0.6212 |        0.6991 |        0.7991 |
| TextRank(0.9) Summarizer (BERT) - bert                                                   |        0.6212 |        0.6993 |        0.7990 |
| TextRank(0.95) Summarizer (BERT) - bert                                                  |        0.6211 |        0.6993 |        0.7988 |
| LexRank Summarizer - bert                                                                |        0.6291 |        0.6895 |        0.7948 |
| LexRankPure Summarizer - bert                                                            |        0.6704 |        0.7148 |        0.8186 |
| TFIDF Summarizer (tf=binary, idf=smooth, tokenizer=bert)                                 |        0.6782 |        0.7593 |        0.8507 |
| TFIDF Summarizer (tf=binary, idf=probabilistic, tokenizer=bert)                          |        0.6707 |        0.7376 |        0.8344 |
| TFIDF Summarizer (tf=raw, idf=smooth, tokenizer=bert)                                    |        0.6767 |        0.7588 |        0.8509 |
| TFIDF Summarizer (tf=raw, idf=probabilistic, tokenizer=bert)                             |        0.6624 |        0.7241 |        0.8225 |
| TFIDF Summarizer (tf=freq, idf=smooth, tokenizer=bert)                                   |        0.5340 |        0.6729 |        0.7885 |
| TFIDF Summarizer (tf=freq, idf=probabilistic, tokenizer=bert)                            |        0.5265 |        0.6652 |        0.7834 |
| TFIDF Summarizer (tf=log_norm, idf=smooth, tokenizer=bert)                               |        0.6755 |      **0.7595** |      0.8504 |
| TFIDF Summarizer (tf=log_norm, idf=probabilistic, tokenizer=bert)                        |        0.6609 |        0.7270 |        0.8261 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.25, idf=smooth, tokenizer=bert)        |        0.6708 |        0.7483 |        0.8447 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.5, idf=smooth, tokenizer=bert)         |        0.6780 |        0.7539 |        0.8479 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.75, idf=smooth, tokenizer=bert)        |        0.6799 |        0.7589 |        0.8508 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.25, idf=probabilistic, tokenizer=bert) |        0.6410 |        0.7186 |        0.8204 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.5, idf=probabilistic, tokenizer=bert)  |        0.6559 |        0.7270 |        0.8265 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.75, idf=probabilistic, tokenizer=bert) |        0.6651 |        0.7342 |        0.8312 |


#### Release 0.16
| Method & Tokenizer                                                                       |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|------------------------------------------------------------------------------------------|---------------|---------------|---------------|
| Random Summarizer - simple                                                               |        0.5566 |        0.6516 |        0.7695 |
| FirstK Summarizer - simple                                                               |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - simple                                                                |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - simple                                                          |        0.6697 |        0.7498 |        0.8433 |
| Band(k=2) Summarizer - simple                                                            |        0.4752 |        0.6275 |        0.7445 |
| Band(k=3) Summarizer - simple                                                            |        0.4911 |        0.6285 |        0.7526 |
| Band(k=6) Summarizer - simple                                                            |        0.4944 |        0.6418 |        0.7601 |
| Rouge1 Summarizer (precision) - simple                                                   |        0.4924 |        0.6298 |        0.7647 |
| Rouge1 Summarizer (recall) - simple                                                      |        0.6726 |        0.7558 |        0.8482 |
| Length Summarizer (char) - simple                                                        |        0.6753 |        0.7577 |        0.8502 |
| Length Summarizer (token) - simple                                                       |        0.6805 |        0.7575 |        0.8510 |
| Random Summarizer - bert                                                                 |        0.5497 |        0.6587 |        0.7744 |
| FirstK Summarizer - bert                                                                 |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - bert                                                                  |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - bert                                                            |        0.6833 |        0.7574 |        0.8488 |
| Band(k=2) Summarizer - bert                                                              |        0.4752 |        0.6275 |        0.7445 |
| Band(k=3) Summarizer - bert                                                              |        0.4911 |        0.6285 |        0.7526 |
| Band(k=6) Summarizer - bert                                                              |        0.4944 |        0.6418 |        0.7601 |
| Rouge1 Summarizer (precision) - bert                                                     |        0.5295 |        0.6500 |        0.7748 |
| Rouge1 Summarizer (recall) - bert                                                        |        0.6851 |        0.7585 |        0.8488 |
| Length Summarizer (char) - bert                                                          |        0.6843 |        0.7588 |        0.8483 |
| Length Summarizer (token) - bert                                                         |        0.6856 |        0.7584 |        0.8520 |
| KMeans Summarizer - bert                                                                 |        0.6599 |        0.7434 |        0.8344 |
| AutoKMeans Summarizer - bert                                                             |        0.6608 |        0.7418 |        0.8333 |
| DecomposedKMeans Summarizer - bert                                                       |        0.6579 |        0.7440 |        0.8341 |
| TextRank(0.05) Summarizer (BERT) - bert                                                  |        0.6212 |        0.7010 |        0.8000 |
| TextRank(0.15) Summarizer (BERT) - bert                                                  |        0.6226 |        0.7004 |        0.7999 |
| TextRank(0.30) Summarizer (BERT) - bert                                                  |        0.6223 |        0.7001 |        0.7999 |
| TextRank(0.5) Summarizer (BERT) - bert                                                   |        0.6232 |        0.7005 |        0.7998 |
| TextRank(0.6) Summarizer (BERT) - bert                                                   |        0.6213 |        0.6993 |        0.7993 |
| TextRank(0.7) Summarizer (BERT) - bert                                                   |        0.6212 |        0.6991 |        0.7991 |
| TextRank(0.85) Summarizer (BERT) - bert                                                  |        0.6212 |        0.6991 |        0.7991 |
| TextRank(0.9) Summarizer (BERT) - bert                                                   |        0.6212 |        0.6993 |        0.7990 |
| TextRank(0.95) Summarizer (BERT) - bert                                                  |        0.6211 |        0.6993 |        0.7988 |
| TFIDF Summarizer (tf=binary, idf=smooth, tokenizer=bert)                                 |        0.6782 |        0.7593 |        0.8507 |
| TFIDF Summarizer (tf=binary, idf=probabilistic, tokenizer=bert)                          |        0.6707 |        0.7376 |        0.8344 |
| TFIDF Summarizer (tf=raw, idf=smooth, tokenizer=bert)                                    |        0.6797 |        **0.7646** |        0.8536 |
| TFIDF Summarizer (tf=raw, idf=probabilistic, tokenizer=bert)                             |        0.5858 |        0.6874 |        0.7941 |
| TFIDF Summarizer (tf=freq, idf=smooth, tokenizer=bert)                                   |        0.6797 |        **0.7646** |        0.8536 |
| TFIDF Summarizer (tf=freq, idf=probabilistic, tokenizer=bert)                            |        0.5858 |        0.6874 |        0.7941 |
| TFIDF Summarizer (tf=log_norm, idf=smooth, tokenizer=bert)                               |        **0.6920** |        0.7642 |        **0.8540** |
| TFIDF Summarizer (tf=log_norm, idf=probabilistic, tokenizer=bert)                        |        0.5344 |        0.6437 |        0.7644 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.25, idf=smooth, tokenizer=bert)        |        0.6777 |        0.7591 |        0.8504 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.5, idf=smooth, tokenizer=bert)         |        0.6782 |        0.7593 |        0.8507 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.75, idf=smooth, tokenizer=bert)        |        0.6782 |        0.7593 |        0.8507 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.25, idf=probabilistic, tokenizer=bert) |        0.6698 |        0.7355 |        0.8330 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.5, idf=probabilistic, tokenizer=bert)  |        0.6704 |        0.7372 |        0.8338 |
| TFIDF Summarizer (tf=double_norm, double_norm_k=0.75, idf=probabilistic, tokenizer=bert) |        0.6706 |        0.7374 |        0.8343 |
| LexRank Summarizer - bert |        0.6293 |        0.6894 |        0.7948 |
| LexRankPure Summarizer - bert |        0.6709 |        0.7147 |        0.8186 |

#### Release 0.15

| Method & Tokenizer                      |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|-----------------------------------------|---------------|---------------|---------------|
| Random Summarizer - simple              |        0.5566 |        0.6516 |        0.7695 |
| FirstK Summarizer - simple              |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - simple               |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - simple         |        0.6697 |        0.7498 |        0.8433 |
| Rouge1 Summarizer (precision) - simple  |        0.4924 |        0.6298 |        0.7647 |
| Rouge1 Summarizer (recall) - simple     |        0.6726 |        0.7558 |        0.8482 |
| Length Summarizer (char) - simple       |        0.6753 |        0.7577 |        0.8502 |
| Length Summarizer (token) - simple      |        0.6805 |        0.7575 |    0.8510 |
| Random Summarizer - bert                |        0.5497 |        0.6587 |        0.7744 |
| FirstK Summarizer - bert                |        0.5070 |        0.6162 |        0.7429 |
| LastK Summarizer - bert                 |        0.5957 |        0.6908 |        0.7972 |
| Rouge1 Summarizer (f1) - bert           |        0.6833 |        0.7574 |        0.8488 |
| Rouge1 Summarizer (precision) - bert    |        0.5295 |        0.6500 |        0.7748 |
| Rouge1 Summarizer (recall) - bert       |    0.6851 |        0.7585 |        0.8488 |
| Length Summarizer (char) - bert         |        0.6843 |        0.7588 |        0.8483 |
| Length Summarizer (token) - bert        |        **0.6856** |        0.7584 |        **0.8520** |
| KMeans Summarizer - bert                |        0.6599 |        0.7434 |        0.8344 |
| AutoKMeans Summarizer - bert            |        0.6608 |        0.7418 |        0.8333 |
| DecomposedKMeans Summarizer - bert      |        0.6579 |        0.7440 |        0.8341 |
| TextRank(0.05) Summarizer (BERT) - bert |        0.6212 |        0.7010 |        0.8000 |
| TextRank(0.5) Summarizer (BERT) - bert  |        0.6232 |        0.7005 |        0.7998 |
| TFIDF Summarizer - bert                 |        0.6781 |        **0.7592** |        0.8504 |




#### Release 0.14
* Results with simple word tokenizer are now available.
  * Simple word tokenizer is not used with clustering based summarizers
  
| Method & Tokenizer                   |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|--------------------------------------|---------------|---------------|---------------|
| Random Summarizer - simple             |        0.5635 |        0.6649 |        0.7799 |
| FirstK Summarizer - simple             |        0.5033 |        0.6154 |        0.7411 |
| LastK Summarizer - simple              |        0.6048 |        0.6973 |        0.8013 |
| Rouge1 Summarizer (f1) - simple        |        0.6641 |        0.7461 |        0.8399 |
| Rouge1 Summarizer (precision) - simple |        0.4918 |        0.6311 |        0.7649 |
| Rouge1 Summarizer (recall) - simple    |        0.6671 |        0.7517 |        0.8447 |
| Length Summarizer (char) - simple      |        0.6669 |        0.7541 |        0.8469 |
| Length Summarizer (token) - simple     |        0.6715 |        0.7548 |        0.8478 |
| Random Summarizer - bert               |        0.5457 |        0.6513 |        0.7698 |
| FirstK Summarizer - bert               |        0.5033 |        0.6154 |        0.7411 |
| LastK Summarizer - bert                |        0.6048 |        0.6973 |        0.8013 |
| Rouge1 Summarizer (f1) - bert          |        0.6727 |        0.7530 |        0.8447 |
| Rouge1 Summarizer (precision) - bert   |        0.5293 |        0.6504 |        0.7745 |
| Rouge1 Summarizer (recall) - bert      |    **0.6753** |        0.7546 |        0.8452 |
| Length Summarizer (char) - bert        |        0.6751 |    **0.7555** |        0.8458 |
| Length Summarizer (token) - bert       |    **0.6753** |        0.7554 |      **0.8492** |
| KMeans Summarizer - bert               |        0.6569 |        0.7432 |        0.8336 |
| AutoKMeans Summarizer - bert           |        0.6576 |        0.7417 |        0.8324 |
| DecomposedKMeans Summarizer - bert     |        0.6549 |        0.7436 |        0.8331 |



#### Release 0.9
* KMeans
* AutoKMeansSummarizer
* DecomposedKMeansSummarize

| Method                     |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|----------------------------|---------------|---------------|---------------|
| Random                     |        0.5513 |        0.6502 |        0.7679 |
| FirstK                     |        0.5033 |        0.6154 |        0.7411 |
| LastK                      |        0.6048 |        0.6973 |        0.8013 |
| Rouge1 (f1)                |        0.6727 |        0.7530 |        0.8447 |
| Rouge1 (precision)         |        0.5293 |        0.6504 |        0.7745 |
| Rouge1 (recall)            |    **0.6753** |        0.7546 |        0.8452 |
| Length (char)              |        0.6751 |      **0.7555** |      0.8458 |
| Length (token)             |        **0.6753** |        0.7554 |    **0.8492** |
| KMeans                     |        0.6569 |        0.7432 |        0.8336 |
| AutoKMeansSummarizer       |        0.6576 |        0.7417 |        0.8324 |
| DecomposedKMeansSummarizer |        0.6550 |        0.7436 |        0.8331 |