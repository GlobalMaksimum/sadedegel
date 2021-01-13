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

### Turkish Tweet Sentiment Classification

Classifier assigns each Turkish tweet texts into two categories ('POSITIVE', 'NEGATIVE') by using sadedegel built-in pipeline.

#### Loading and Predicting with the Model:

```python
from sadedegel.prebuilt import tweet.sentiment

# We load our prebuilt model:
model = tweet_sentiment.load()

# Here we enter our text to get sentiment predictions.
y_pred = model.predict([])
```

#### Accuracy

Current prebuilt model has 3-fold cross validation accuracy score of 0.791 and having accuracy 0.746 on private test set.