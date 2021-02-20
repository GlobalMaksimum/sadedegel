SadedeGel prebuilt models are built to extend capabilities of sadedegel library 
on common NLP tasks we encounter every day. Such as sentiment analysis, profanity detection.

Open source prebuilt models are not designed to achieve state of the art accuracies. They rather provide a good starting 
point by training sklearn based limited memory (`partial_fit` using micro batches) models with a single pass over training data.

If you need access for our state of the art models, please reach us at info@sadedegel.ai

### Turkish News Text Classification

Classifier assigns each Turkish text into one of 12 categories (`sadedegel.dataset.tscorpus.CATEGORIES`)
by using a sadedegel including pipeline

#### Loading model

```python
from sadedegel.prebuilt import news_classification

model = news_classification.load()

y_pred = model.predict([''])
```

To convert class ids to class labels use

```python
from sadedegel.dataset.tscorpus import CATEGORIES

y_pred_label = [CATEGORIES.index(_y_pred) for _y_pred in y_pred]
```

#### Accuracy

Current prebuilt model has an average class prediction cv-3 accuracy of `0.746`

### Turkish Tweets Profanity Classification

This classifier assigns Turkish tweets to one of `OFF`, `NOT` classes based on whether a tweet contains a profane language or not, by using a `sadedegel`pipeline.

#### Load Model
```python
from sadedegel.prebuilt import tweet_profanity

model = tweet_profanity.load()  # Load latest version

y_pred = model.predict(['bir takım ağza alınmayacak sözcükler.'])
```
To convert predictions to profanity label by class mapping:

```python
from sadedegel.prebuilt.tweet_profanity import load, CLASS_VALUES
model = load()

y_pred = model.predict(['bir takım ağza alınmayacak sözcükler.'])

y_pred_value = [CLASS_VALUES[_y_pred] for _y_pred in y_pred]
```

#### Accuracy

Current prebuilt tweet profanity model has an **macro-F1** score of `0.6619` on test set.
> Best model in [SemEval-2020 Task 12](https://arxiv.org/pdf/2006.07235.pdf) has `0.8258` accuracy

### Turkish Tweet Sentiment Classification
Classifier assigns each Turkish tweet texts into two classes ('POSITIVE', 'NEGATIVE') by using sadedegel built-in pipeline.

#### Loading and Predicting with the Model:
```python
from sadedegel.prebuilt import tweet_sentiment
# We load our prebuilt model:
model = tweet_sentiment.load()

# Here we enter our text to get sentiment predictions.
y_pred = model.predict([])
```
#### Accuracy

Current prebuilt model has 
* 3-fold cross validation F1 macro score of `mean 0.7946, std 0.0043)`.
* 5-fold cross validation F1 macro score of `mean 0.7989, std 0.0055)` 