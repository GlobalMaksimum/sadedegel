import json
import numpy as np
import lightgbm as lgb
from os import makedirs
from pathlib import Path
from functools import partial
import warnings
import joblib
from rich.console import Console
from rich.live import Live

warnings.filterwarnings("ignore")

console = Console()

try:
    import pandas as pd
except ImportError:
    console.log(("pandas package is not a general sadedegel dependency."
                 " But we do have a dependency on building our supervised ranker model"))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARN)
except ImportError:
    console.log(("optuna package is not a general sadedegel dependency."
                 " But we do have a dependency on tuning our supervised ranker model. Please install optuna to proceed."))


def check_log_dir(data_home="~/.sadedegel_data"):
    logs_path = Path(data_home).expanduser() / "logs"
    if not logs_path.exists():
        makedirs(logs_path)


def create_json_if_notxsts(data_home="~/.sadedegel_data", json_name=None):
    logs_path = Path(data_home).expanduser() / "logs"
    if not (logs_path / json_name).exists():
        with open(str(logs_path / json_name), "w") as jfile:
            json.dump({}, jfile)


def log_early_stop(trial, mean_rounds, run_name):
    check_log_dir()
    create_json_if_notxsts(json_name=f"{run_name}_early_stop.json")
    path = Path(f"~/.sadedegel_data/logs/{run_name}_early_stop.json").expanduser()
    with open(path, "r+") as jfile:
        trial_round_dict = json.load(jfile)
        trial_round_dict.update({trial: mean_rounds})
        jfile.seek(0)
        json.dump(trial_round_dict, jfile, indent=4)


def log_best_params(study, run_name):
    check_log_dir()
    path = Path(f"~/.sadedegel_data/logs/{run_name}_best_trial.json").expanduser()

    best_trial_dict = dict()
    best_trial_dict["params"] = study.best_params
    best_trial_dict["score"] = study.best_value
    best_trial_dict["trial"] = study.best_trial.number

    console.log(f"Optimization DONE. Best Score so far: {study.best_value}. Saving parameter space to ~/.sadedegel_data/logs")

    with open(path, "w") as jfile:
        json.dump(best_trial_dict, jfile)


def parse_early_stop(run_name, best_trial):
    path = Path(f"~/.sadedegel_data/logs/{run_name}_early_stop.json").expanduser()
    with open(path, "r") as jfile:
        trials_dict = json.load(jfile)
    return int(trials_dict[str(best_trial)])


def parse_best_trial(run_name):
    path = Path(f"~/.sadedegel_data/logs/{run_name}_best_trial.json").expanduser()
    with open(path, "r") as jfile:
        best_params_dict = json.load(jfile)

    return best_params_dict["params"], best_params_dict["trial"]


def ranker_objective(trial, vectors, metadata, k, run_name, live):

    if trial.number == 0:
        live.update("Optuna tuning has started. Trial results will be reported live...", refresh=True)

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.9),
        "num_leaves": trial.suggest_int("num_leaves", 4, 512),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000),
        "max_bin": trial.suggest_int("max_bin", 150, 300),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-3, 5),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-3, 5),
        "bagging_fraction": trial.suggest_loguniform("bagging_fraction", 0.6, 0.95),
        "feature_fraction": trial.suggest_loguniform("feature_fraction", 0.3, 0.95),
    }

    summarization_perf = []
    best_iters = []
    uniq_docs = metadata.doc_id.unique()

    # Leave-one-out cross validation over documents.
    for doc_id in uniq_docs:
        train_docs = metadata.loc[metadata.doc_id != doc_id]
        valid_docs = metadata.loc[metadata.doc_id == doc_id]

        train_ixs = train_docs.index.tolist()
        valid_ixs = valid_docs.index.tolist()

        train_X, train_y = vectors[train_ixs], train_docs.relevance.values
        valid_X, valid_y = vectors[valid_ixs], valid_docs.relevance.values

        qids_train = train_docs.groupby("doc_id")["doc_id"].count().to_numpy()
        qids_valid = valid_docs.groupby("doc_id")["doc_id"].count().to_numpy()

        eval_at_perc = int(valid_X.shape[0] * k)

        # Fit ranker model
        ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", verbose=-100, **param_grid)
        ranker.fit(X=train_X,
                   y=train_y,
                   group=qids_train,
                   eval_set=[(valid_X, valid_y)],
                   eval_group=[qids_valid],
                   eval_at=eval_at_perc,
                   verbose=-100,
                   callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])

        # Collect best early stopping for each validation document
        best_iters.append(ranker.best_iteration_)
        summarization_perf.append(ranker.best_score_["valid_0"][f"ndcg@{eval_at_perc}"])

    log_early_stop(trial=trial.number,
                   mean_rounds=np.mean(best_iters),
                   run_name=run_name)

    return np.mean(summarization_perf)


def live_update_callback(study, trial, total_trials, live):
    live.update(f"Trial: {trial.number + 1}/{total_trials} - Trial Score: {trial.value} - Best Score So Far: {study.best_value}", refresh=True)


def optuna_handler(n_trials, run_name, metadata, vectors, k):
    with Live(console=console, screen=True, auto_refresh=False) as live:
        objective = partial(ranker_objective, run_name=run_name, metadata=metadata, vectors=vectors, k=k, live=live)
        live_update = partial(live_update_callback, total_trials=n_trials, live=live)
        study = optuna.create_study(direction="maximize", study_name="LGBM Ranker")
        study.optimize(objective, n_trials=n_trials, callbacks=[live_update])

    log_best_params(study, run_name=run_name)


def create_empty_model(run_name: str):
    params, trial = parse_best_trial(run_name=run_name)
    num_rounds = parse_early_stop(run_name=run_name, best_trial=trial)
    params["n_estimators"] = num_rounds

    model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", verbose=-100, **params)

    return model


def fit_ranker(ranker: lgb.LGBMRanker, vectors: np.ndarray, metadata: pd.DataFrame):
    console.log("Fitting model with optimal parameter space.", style="cyan")

    train_X, train_y = vectors, metadata.relevance.values
    qids_train = metadata.groupby("doc_id")["doc_id"].count().to_numpy()
    ranker.fit(train_X, train_y, group=qids_train)

    return ranker


def save_ranker(ranker: lgb.LGBMRanker, name: str):
    basepath = Path(f"~/.sadedegel_data/models").expanduser()
    if not basepath.exists():
        makedirs(basepath)
    path = Path(f"{basepath}/ranker_{name}.joblib").expanduser()
    joblib.dump(ranker, path)

    console.log(f"Model saved to ~/.sadedegel_data/models with name ranker_{name}")
