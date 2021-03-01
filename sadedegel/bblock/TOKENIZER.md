Evaluation of built in tokenizers are made using TsCorpus (`sadedgel.dataset.tscorpus`)

## Performance (doc/sec)

Performance of sadedegel tokenizers are given as below

| Tokenizer       |   doc/sec |   
|-----------------|---------------|
| bert            |      >225 doc/sec | 
| simple          |      >545 doc/sec   |
| icu             |      >1300 doc/sec   |

## Jaccard Similarity (IoU) Metric

| Tokenizer       |   IoU (macro) |   IoU (micro) |   
|-----------------|---------------|---------------|
| bert            |      0.4592   | 0.4439        |  
| simple          |      0.8544   | 0.8668        |
| icu             |      0.9594   | 0.9608        |

## Weighted Jaccard Similarity

Given that list produced by a tokenizer is a multi-set (allowing same words to occur more than once), so a fair
comparison should take number of occurrence into
account ([weighted jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance))

| Tokenizer       |   IoU (macro)|   IoU (micro) |   
|-----------------|--------------|---------------|
| bert            |    0.4884    | 0.4860        |  
| simple          |    0.7819    | 0.7791        |
| icu             |    0.9501    | 0.9472        |

Results can be reproduced by using

`python -m sadedegel.bblock.cli tokenizer-evaluate`   