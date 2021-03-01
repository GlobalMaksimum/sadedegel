import numpy as np
from scipy.stats.distributions import uniform
from sklearn.model_selection import ParameterSampler


def bm25_parameter_space(n_trials):
    rng = np.random.RandomState(42)

    return ParameterSampler(
        dict(tf_method=["binary", "raw", "freq", "log_norm", "double_norm"]
             , idf_method=["smooth", "probabilistic"],
             drop_stopwords=[True, False],
             drop_suffix=[True, False], drop_punct=[True, False],
             lowercase=[True, False],
             k1=uniform(1.2, 2.0),
             b=uniform(0.5, 0.8), delta=uniform(0, 2)), n_iter=n_trials, random_state=rng)


def tfidf_parameter_space(n_trials):
    rng = np.random.RandomState(42)

    return ParameterSampler(
        dict(tf_method=["binary", "raw", "freq", "log_norm", "double_norm"]
             , idf_method=["smooth", "probabilistic"],
             drop_stopwords=[True, False],
             drop_suffix=[True, False], drop_punct=[True, False],
             lowercase=[True, False]), n_iter=n_trials, random_state=rng)


def lexrank_parameter_space(n_trials):
    rng = np.random.RandomState(42)

    return ParameterSampler(
        dict(tf_method=["binary", "raw", "freq", "log_norm", "double_norm"]
             , idf_method=["smooth", "probabilistic"],
             drop_stopwords=[True, False],
             drop_suffix=[True, False], drop_punct=[True, False],
             lowercase=[True, False], threshold=uniform(0, 1)), n_iter=n_trials, random_state=rng)


def textrank_parameter_space(n_trials):
    rng = np.random.RandomState(42)

    return ParameterSampler(
        dict(alpha=uniform(0.0001, 1)), n_iter=n_trials, random_state=rng)
