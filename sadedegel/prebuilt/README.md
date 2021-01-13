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

### Customer Sentiment Analysis

Classifier assigns each customer product review into one of `POSITIVE`, `NEGATIVE`, `NEUTRAL` categories.

#### Loading model
```python
from sadedegel.prebuilt import customer_sentiment

model = customer_sentiment.load() # Loads latest version

y_pred = model.predict(['product review text.'])
```

To convert predictions to sentiment by class mapping:

```python
y_pred_sentiment = [customer_sentiment._classes[pred] for pred in predictions]

```

#### Accuracy

Current prebuilt customer sentiment model has an accuracy of `0.685` on an unseen evaluation set.