## SadedeGel Model

We model a human annotator for summarization task as follows:
1. A human annotator reads a news document and 
choose to drop out some sentences in it (any number in any round) based on his/her prior knowledge of language and news' domain.
2. Annotator reads left over document again and repeat 1. until no sentences are left in the news document.

We keep track of human annotator behaviour with [SadedeGel Annotator](https://github.com/GlobalMaksimum/sadedegel-annotator) 
by recoding the **Round** of each sentences in which it is eliminated.

Later a sentence is eliminated, higher its relative score is within a given news document. 

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