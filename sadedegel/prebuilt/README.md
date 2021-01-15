SadedeGel prebuilt models are built to extend capabilities to address common NLP tasks we encounter every day.

They are currently experimental stage but will grow in size and accuracy in time

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

This classifier assigns Turkish tweets to one of `PROFANE`, `PROPER` labels based on whether a tweet contains a profane language or not by using a `sadedegel`pipeline.

#### Load Model
```python
from sadedegel.prebuilt import tweet_profanity

model = tweet_profanity.load()  # Load latest version

y_pred = model.predict(['bir takım ağza alınmayacak sözcükler.'])
``` 
To convert predictions to profanity label by class mapping:

```python
y_pred_profanity = [tweet_profanity._classes[pred] for pred in predictions]
```

#### Accuracy

Current prebuilt tweet profanity model has an **ROC-AUC** score of `0.824` on 3-fold CV.