import optuna
from .customer_sentiment import empty_model
from pathlib import Path
from os.path import dirname

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from ..extension.sklearn import TfidfVectorizer, OnlinePipeline

data_dir = Path(dirname(__file__)) / 'data' / 'customer_satisfaction' / 'customer_train.csv.gz'
df = pd.read_csv(data_dir)
X = TfidfVectorizer(tf_method='freq', idf_method='smooth', drop_stopwords=False, lowercase=False,
                    drop_suffix=False).fit_transform(df.text)
y = df.sentiment


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 5, log=True)

    pipeline = OnlinePipeline([
        ("nb_model", MultinomialNB(alpha=alpha))
    ])

    score = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    accuracy = score.mean()
    return accuracy


def evaluate():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(study.best_trial)


if __name__ == '__main__':
    evaluate()
