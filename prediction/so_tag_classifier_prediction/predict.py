import argparse

import joblib
from so_tag_classifier_core import text_prepare, tokenize_and_stem

model = None
mlb = None


def prepare_model_if_needed(fname: str) -> None:
    """ Loads the model if it's not already in memory. """
    global model, mlb
    if model is None and mlb is None:
        model, mlb = joblib.load(fname)


def predict(sentence: str, fname: str) -> list:
    """ Makes a label prediction based on a sentence and a model"""
    global model, mlb
    prepare_model_if_needed(fname)

    pred = model.predict([sentence])
    return mlb.inverse_transform(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a multi-label StackOverflow tag set for a list of texts"
    )
    parser.add_argument("-t", "--txt", type=str, help="a single text to predict on")
    parser.add_argument(
        "-mp", "--model_path", type=str, help="the model path to the model (.pkl file) to load from local"
    )

    args = parser.parse_args()
    fname = args.model_path

    import time

    start = time.time()
    print(predict(args.txt, fname))
    print(f"predicted in {time.time() - start} seconds")
