import argparse
import importlib.resources as pkg_resources
import tempfile

import boto3
import joblib
import yaml
from so_tag_classifier_core import _TAGS_TO_KEEP, text_prepare, tokenize_and_stem

_AWS_CONFIGS = yaml.safe_load(pkg_resources.open_text("so_tag_classifier_prediction", "aws_config.yml"))
_BUCKET = _AWS_CONFIGS.get("bucket")

model = None
mlb = None


def load_aws_config(fname: str):
    """
    Helper function to load Bucket name from a yaml file
    """


def load_model_from_s3(s3_key: str):
    """
    Returns model loaded from S3
    """
    global _BUCKET
    s3 = boto3.client("s3")
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=_BUCKET, Key=s3_key)
        fp.seek(0)
        model = joblib.load(fp)
    return model


def prepare_model_if_needed(s3_key: str) -> None:
    """ Loads the model if it's not already in memory. """
    global model, mlb
    if model is None and mlb is None:
        model, mlb = load_model_from_s3(s3_key)


def get_preds(model, binarizer, sentence: str) -> set:
    """Get labels of prediction for a single sentence (hence the 0)"""
    raw_preds = model.predict([sentence])
    return binarizer.inverse_transform(raw_preds)[0]


def get_probability_preds(model, sentence: str) -> dict:
    """Get probabilities for all tags possible for a single sentence (hence the 0)"""
    global _TAGS_TO_KEEP
    probs = model.predict_proba([sentence])[0]
    return dict(zip(_TAGS_TO_KEEP, probs))


def predict(sentence: str, s3_key: str) -> dict:
    """ Makes a label prediction with probability, based on a sentence and a model"""
    global model, mlb
    prepare_model_if_needed(s3_key)

    preds = get_preds(model, mlb, sentence)
    probs_dict = get_probability_preds(model, sentence)
    return dict(zip(preds, [probs_dict.get(key) for key in preds]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a multi-label StackOverflow tag set for a list of texts"
    )
    parser.add_argument("-t", "--txt", type=str, help="a single text to predict on")
    parser.add_argument(
        "-mp", "--model_path", type=str, help="the model path to the model (.pkl file) to load from local"
    )

    args = parser.parse_args()
    s3_key = args.model_path

    import time

    start = time.time()
    print(predict(args.txt, s3_key))
    print(f"predicted in {time.time() - start} seconds")
