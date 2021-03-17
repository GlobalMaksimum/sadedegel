# Tokenizer Performance and Accuracy

Built in tokenizers are evaluated on TsCorpus (`sadedegel.dataset.tscorpus`) dataset.

## Performance (doc/sec)

Performance of sadedegel tokenizers are given as below

| Tokenizer       |   doc/sec |   
|-----------------|---------------|
| bert            |      >167 doc/sec | 
| simple          |      >545 doc/sec   |
| icu             |      **>1300 doc/sec**   |

## Jaccard Similarity (IoU) Metric

| Tokenizer       |   IoU (macro) |   IoU (micro) |   
|-----------------|---------------|---------------| 
| simple          |      0.8544   | 0.8668        |
| bert            |       0.8739  | 0.8860        |
| icu             |      **0.9594**   | **0.9608**        |

## Weighted Jaccard Similarity

Given that, list produced by a tokenizer is a multi-set (allowing same token type to repeat more than once), a fair
comparison should take number of word type occurrences into
account ([weighted jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance))

| Tokenizer       |   IoU (macro)|   IoU (micro) |   
|-----------------|--------------|---------------|
| bert            |    -    | -        |  
| simple          |    0.7819    | 0.7791        |
| icu             |    **0.9501**    | **0.9472**        |

### Reproducibility

Results can be reproduced using

```bash
python -m sadedegel.bblock.cli tokenizer-evaluate
```   