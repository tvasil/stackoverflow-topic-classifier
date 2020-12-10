import argparse
import datetime
import logging
import warnings

import boto3
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
_BUCKET = "tvasil-ml-models"


def read_data(fname):
    df = pd.read_csv(args.train_data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["tags"].values, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def _get_training_pipeline(X_train, y_train_binarized):
    """
    Build the training pipe based on some default configuration.
    TODO: Externalize the parameters of this default configuration to a file
    """

    estimators = [
        ("preprocessor", FunctionTransformer(text_prepare, kw_args={"join_symbol": " "})),
        (
            "tfidf",
            TfidfVectorizer(tokenizer=tokenize_and_stem, ngram_range=(1, 3), norm="l2", max_df=0.9, min_df=5),
        ),
        ("clf", ClassifierChain(LogisticRegression(C=10, penalty="l1", dual=False, solver="liblinear"))),
    ]

    return Pipeline(estimators, verbose=True)


def _fit(model, X_train, y_train_binarized):
    """
    Wrapper for fitting to modify metrics we want to capture in MLFlow
    """
    model.fit(X_train, y_train_binarized)
    return model


def train_gridsearch_crossval(X_train, y_train_binarized, search_space):
    """
    Trains a model with crossvalidation and RandomizedSearchCV, which might take a while, as at least 150 folds are expored
    """

    training_pipe = _get_training_pipeline(X_train, y_train_binarized)

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
    rs = _fit(model=rs, X_train=X_train, y_train_binarized=y_train_binarized)

    return rs.best_estimator_


def train_base(X_train, y_train_binarized):
    """
    Simple model training using a pipeline or any other model setup
    """
    pipeline = _get_training_pipeline(X_train, y_train_binarized)
    model = _fit(model=pipeline, X_train=X_train, y_train_binarized=y_train_binarized)
    return model


def save_in_s3(fname: str) -> None:
    """
    Saves the resulting model + ybinarizer as a pickle and then uploads to S3
    """
    s3 = boto3.resource("s3")
    s3.Object(_BUCKET, fname).put(Body=open("tmp/" + fname, "rb"))


def _get_model_filename(model, binarizer):
    """
    Produce the model filename based on some logic
    """
    sha_model = joblib.hash(model, hash_name="sha1")
    sha_binarizer = joblib.hash(binarizer, hash_name="sha1")
    return sha_model + sha_binarizer + ".pkl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label StackOverflow tag prediction model")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="the full path of the .csv training data; columns should be 'text' and 'tags'",
    )
    parser.add_argument(
        "--logger_file",
        type=str,
        required=True,
        help="name of file where we'll log the training process and results",
    )
    parser.add_argument(
        "--gs",
        default=False,
        action="store_true",
        help="whether gridsearch with cross validation should be performed",
    )

    args = parser.parse_args()

    logger = logging.getLogger("model_training")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logger_file))
    logger.addHandler(logging.StreamHandler())

    X_train, X_test, y_train, y_test = read_data(args.train_data)
    binarizer, y_train_binarized, y_test_binarized = binarize_ys(y_train, y_test)

    if bool(args.gs):
        model = train_gridsearch_crossval(X_train, y_train_binarized, search_space)
    else:
        model = train_base(X_train, y_train_binarized)

    import sys

    log_performance(X_test, y_test_binarized, model, binarizer, logger)

    fname = _get_model_filename(model, binarizer)
    joblib.dump((model, binarizer), "tmp/" + fname)
    save_in_s3(fname)  # upload to S3
    logger.info("Finished training! Check the files and model in S3")
