Evaluation of built in tokenizers are made using TsCorpus (`sadedgel.dataset.tscorpus`)


| Tokenizer       |   IoU (macro) |   IoU (micro) |   
|-----------------|---------------|---------------|
| bert            |        0.8604 | 0.8702   |  
| simple          |      0.8314   | 0.8395   |

Results can be reproduce by using 


`python -m sadedegel.bblock tokenizer-evaluate`   