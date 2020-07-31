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

#### Release 0.9
* KMeans
* AutoKMeansSummarizer
* DecomposedKMeansSummarize

| Method                     |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|----------------------------|---------------|---------------|---------------|
| Random                     |      0.551282 |      0.650239 |      0.767942 |
| FirstK                     |      0.503327 |      0.615367 |      0.741094 |
| LastK                      |      0.604835 |      0.697297 |      0.801343 |
| Rouge1 (f1)                |      0.67267  |      0.752952 |      0.844675 |
| Rouge1 (precision)         |      0.529314 |      0.650443 |      0.774472 |
| Rouge1 (recall)            |      0.675343 |      0.754626 |      0.845244 |
| KMeans                     |      0.65688  |      0.743172 |      0.833607 |
| AutoKMeansSummarizer       |      0.657585 |      0.741666 |      0.832376 |
| DecomposedKMeansSummarizer |      0.655008 |      0.743582 |      0.833087 |