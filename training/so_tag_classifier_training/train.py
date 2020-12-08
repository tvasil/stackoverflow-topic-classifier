import argparse
import datetime
import logging
import warnings

import joblib
import pandas as pd
from params.parameter_tuning import scoring, search_space
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from so_tag_classifier_core.evaluation_logs import log_performance
from so_tag_classifier_core.preprocessing_steps import binarize_ys, text_prepare, tokenize_and_stem

warnings.filterwarnings("ignore", message=r"unknown class.*")

_TEST_SIZE = 0.1
_RANDOM_STATE = 42
_RS_N_ITER = 50
_RS_CV = 3


def read_data(fname):
    df = pd.read_csv(args.train_data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["tags"].values, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label StackOverflow tag prediction model")
    parser.add_argument(
        "train_data",
        type=str,
        help="the full path of the .csv training data; columns should be 'text' and 'tags'",
    )
    parser.add_argument(
        "logger_file", type=str, help="name of file where we'll log the training process and results"
    )

    args = parser.parse_args()

    logger = logging.getLogger("model_training")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logger_file))
    logger.addHandler(logging.StreamHandler())

    X_train, X_test, y_train, y_test = read_data(args.train_data)
    binarizer, y_train_binarized, y_test_binarized = binarize_ys(y_train, y_test)

    estimators = [
        ("preprocessor", FunctionTransformer(text_prepare, kw_args={"join_symbol": " "})),
        (
            "tfidf",
            TfidfVectorizer(tokenizer=tokenize_and_stem, ngram_range=(1, 3), norm="l2", max_df=0.9, min_df=5),
        ),
        ("clf", ClassifierChain(LogisticRegression(C=10, penalty="l1", dual=False, solver="liblinear"))),
    ]

    training_pipe = Pipeline(estimators, verbose=True)
    rs = RandomizedSearchCV(
        training_pipe,
        param_distributions=search_space,
        scoring=scoring,
        refit="f1",
        return_train_score=True,
        n_iter=_RS_N_ITER,
        cv=_RS_CV,
        verbose=10,
        n_jobs=-1,
    )

    rs.fit(X_train, y_train_binarized)

    log_performance(
        X_test=X_test, y_test_binarized=y_test_binarized, model=rs, binarizer=binarizer, logger=logger
    )

    fname = "models/" + str(datetime.date.today()).replace("-", "_") + "_rs_model_and_mlb.pkl"
    joblib.dump((rs.best_estimator_, binarizer), fname)
