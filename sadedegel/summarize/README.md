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

### Performance Evaluation

Users can simply perform contrastive evaluation of the implemented summarizers running `sadedegel-summarize evaluate` on command line. 

##### Options: 

- `--tag | -t`: Each implemented summarizer has a built-in "tag" feature from the list `[extractive|baseline|self-supervised|ml]`. 
User can provide subsets to include specific  summarizers and exclude the rest.
- `--word-tokenizer | -wt`: Sadedegel incorporates two word tokenizers `bert` and `simple`. Default word tokenizer `bert` can be overriden for evaluating performance of summarizers subject to other built-in or user-defined word tokenizers.
- `--debug | -db`: Set to `True` when evaluation CLI is tested with SadedeGel test suite. Just as new features are tested, their evaluation is also subject to unit tests. Addition of new summarizers or tags require the validation of the evaluation script as well. Refer to [CONTRIBUTING.md](https://github.com/GlobalMaksimum/sadedegel/blob/develop/CONTRIBUTING.md) for more on adding tests.  
### Performance Table

#### Release 0.9
* KMeans
* AutoKMeansSummarizer
* DecomposedKMeansSummarize
* Backend word tokenizer: `BertTokenizer`

| Method                     |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|----------------------------|---------------|---------------|---------------|
| Random                     |        0.5513 |        0.6502 |        0.7679 |
| FirstK                     |        0.5033 |        0.6154 |        0.7411 |
| LastK                      |        0.6048 |        0.6973 |        0.8013 |
| Rouge1 (f1)                |        0.6727 |        0.7530 |        0.8447 |
| Rouge1 (precision)         |        0.5293 |        0.6504 |        0.7745 |
| Rouge1 (recall)            |    **0.6753** |        0.7546 |        0.8452 |
| Length (char)              |        0.6751 |      **0.7555** |      0.8458 |
| Length (token)             |        0.6753 |        0.7554 |    **0.8492** |
| KMeans                     |        0.6569 |        0.7432 |        0.8336 |
| AutoKMeansSummarizer       |        0.6576 |        0.7417 |        0.8324 |
| DecomposedKMeansSummarizer |        0.6550 |        0.7436 |        0.8331 |

#### Release 0.13.5
* Backend word tokenizer: `SimpleTokenizer`

| Method                        |   ndcg(k=0.1) |   ndcg(k=0.5) |   ndcg(k=0.8) |
|-------------------------------|---------------|---------------|---------------|
| Random Summarizer             |        0.5635 |        0.6649 |        0.7799 |
| FirstK Summarizer             |        0.5033 |        0.6154 |        0.7411 |
| LastK Summarizer              |        0.6048 |        0.6973 |        0.8013 |
| Rouge1 Summarizer (f1)        |        0.6641 |        0.7461 |        0.8399 |
| Rouge1 Summarizer (precision) |        0.4918 |        0.6311 |        0.7649 |
| Rouge1 Summarizer (recall)    |        0.6671 |        0.7517 |        0.8447 |
| Length Summarizer (char)      |        0.6669 |        0.7541 |        0.8469 |
| Length Summarizer (token)     |        **0.6715** |        **0.7548** |        **0.8478** |
| KMeans Summarizer             |        0.6569 |        0.7432 |        0.8336 |
| AutoKMeans Summarizer         |        0.6576 |        0.7417 |        0.8324 |
| DecomposedKMeans Summarizer   |        0.6555 |        0.7436 |        0.8331 |


