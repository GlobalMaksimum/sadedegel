from functools import partial

import optuna
import pandas as pd
import sklearn
from rich.console import Console
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sadedegel.dataset.telco_sentiment import load_telco_sentiment_train
from sadedegel.extension.sklearn import Text2Doc, HashVectorizer, CharHashVectorizer

console = Console()


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective_partial(trial, X, y):
    n_feature = trial.suggest_int("n_features", 100, 1048576)
    alternate_sign = trial.suggest_categorical("alternate_sign", [True, False])
    ngram_range_opt = trial.suggest_categorical("ngram_range", ["T12", "T13", "T25", "L35", "T35",
                                                                "L13579", "L12468", "T24"])

    if ngram_range_opt[0] == "T":
        ngram_range = (int(ngram_range_opt[1]), int(ngram_range_opt[2]))
    elif ngram_range_opt[0] == "L":
        ngram_range = [int(n) for n in ngram_range_opt[1:]]

    console.log(f"ngram_range: {ngram_range}")
    console.log(f"n_features: {n_feature}")
    console.log(f"alternate_sign: {alternate_sign}")

    X = CharHashVectorizer(n_feature, ngram_range=ngram_range, alternate_sign=alternate_sign).transform(X)

    classifier = trial.suggest_categorical("classifier", ["sgd", "svc"])
    console.log(f"Classifier: {classifier}")

    if classifier == "sgd":

        sgd_alpha = trial.suggest_float("sgd_alpha", 1e-10, 10, log=True)
        sgd_penalty = trial.suggest_categorical("sgd_penalty", ["elasticnet", "l2"])
        loss = trial.suggest_categorical("loss", ["modified_huber", "log"])

        console.log(f"Alpha: {sgd_alpha}")
        console.log(f"Penalty: {sgd_penalty}")
        console.log(f"Loss: {loss}")

        classifier = SGDClassifier(alpha=sgd_alpha, penalty=sgd_penalty, loss=loss, random_state=42)
    else:
        c = trial.suggest_float("C", 1e-10, 100, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])

        console.log(f"C: {c}")
        console.log(f"kernel: {kernel}")

        if kernel == "rbf":
            gamma = trial.suggest_categorical("gamma", ["scale", "auto", "float"])

            console.log(f"gamma: {gamma}")

            if gamma in ["scale", "auto"]:
                classifier = SVC(C=c, kernel=kernel, gamma="scale", probability=True, random_state=42)
            else:
                gamma0 = trial.suggest_float("gamma0", 1e-10, 100, log=True)

                console.log(f"gamma: {gamma0}")

                classifier = SVC(C=c, kernel=kernel, gamma=gamma0, probability=True, random_state=42)
        elif kernel == "poly":
            d = trial.suggest_categorical("degree", [2, 3])

            console.log(f"degree: {d}")

            classifier = SVC(C=c, kernel=kernel, degree=d, probability=True, random_state=42)
        else:
            classifier = SVC(C=c, kernel=kernel, probability=True, random_state=42)

    score = sklearn.model_selection.cross_val_score(classifier, X, y, scoring="accuracy", n_jobs=-1, cv=5)
    accuracy = score.mean()

    return accuracy


def driver(max_records=-1):
    df = pd.DataFrame.from_records(load_telco_sentiment_train())

    if max_records > 0:
        df = df.sample(n=max_records, random_state=42)

    X = Text2Doc("icu", hashtag=True, emoji=True, mention=True, progress_tracking=True).transform(df.tweet)
    y = df.sentiment_class

    objective = partial(objective_partial, X=X, y=y)

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=50)

    console = Console()
    console.print(study.best_trial)


if __name__ == "__main__":
    driver(-1)

"""
params={'n_features': 413833, 'alternate_sign': False, 'classifier': 'svc', 'C': 0.28610731097622305, 'kernel': 'linear'}
"""
