 # Sentence Tokenizer
 
 Sentence tokenizer (also known as sentence boundary detection(SBD)) is one of the key components of 
 our extraction-based summarizer:
 
 1. Extraction-based algorithm picks up a subset of all sentences in document. In return, it is crucial to detect and split sentences correctly.
 2. 
 
 ## SBD Evaluation
 
 To test existing SBD algorithms use.
 
```bash
python3 -m sadedegel.tokenize evaluate
```

```bash
IoU score for NLTKPunctTokenizer
        Micro IoU: 0.7071
        Macro IoU: 0.7343
IoU score for RegexpSentenceTokenizer
        Micro IoU: 0.6812
        Macro IoU: 0.7224
IoU score for MLBasedTokenizer
        Micro IoU: 0.8881
        Macro IoU: 0.8946
```

Our current best model is a RF based SBD classifier based on features described in [Speech and Language Processing] (Chapter 6)

[Speech and Language Processing]: (https://web.stanford.edu/~jurafsky/slp3/)

`-v` option of `evaluate` will yield contribution of each document to your error in detail.

### SBD Difference


For detailed analysis of errors by `MLBasedTokenizer` use `diff` command

```bash
python3 -m sadedegel.tokenize diff
```

```bash
+ Yaşananlar
+ Savu(r)ma Bİr insan, ‘‘Kendimi savunacağım’’ derken, savurursa ne olur?..
+ SAVUNMASI VAR
+ Bir de, evlere şenlik savunması var beyefendinin.
+ CEVABINI VERİYOR
+ Zat-ı muhteremin yazısının satır araları itiraflarla dolu...
+ KOLTUKLAR YAKIN
+ Gelelim bir diğer meseleye...
+ Ünlü sözler ‘‘Orkestrayı yönetmek isteyen kalabalığa sırtını döner.’’ James Crook

- Yaşananlar  Savu(r)ma Bİr insan, ‘‘Kendimi savunacağım’’ derken, savurursa ne olur?..
- SAVUNMASI VAR Bir de, evlere şenlik savunması var beyefendinin.
- CEVABINI VERİYOR Zat-ı muhteremin yazısının satır araları itiraflarla dolu...
- KOLTUKLAR YAKIN Gelelim bir diğer meseleye...
- Ünlü sözler ‘‘Orkestrayı yönetmek isteyen kalabalığa sırtını döner.’’
```

## Rebuild Model

You can change our feature set or classifier pipeline in anyway you like to obtain better models.

To re-train the existing `sbd` model use

```bash
python3 -m sadedegel.tokenize build
```

