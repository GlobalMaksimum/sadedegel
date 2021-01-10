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