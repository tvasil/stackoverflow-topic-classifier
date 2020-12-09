# stackoverflow-topic-classifier
This Python project contains the code to train, evaluate and predict from a machine learning model that attempts to identify the appropriate <tags> that a StackOverflow post should receive. The idea of the project is the following:

Given some concatentades title + body text of a StackOverflow post such [the following](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python):

```
What are metaclasses in Python?

In Python, what are metaclasses and what do we use them for?
```

can we predict the tags that the user would have given to the post? Specifically, `['python', 'oop', 'metaclass', 'python-datamodel']`

## Requirements

The minimal requirements are:
```
- python 3.8
- scikit-learn 0.23.2
- joblib 0.17
- nltk 3.5
```
For convenience, I've exported the full `environment.yml` file from my conda environment. Mind you, that's just for reproducibility and is not used by each of the sub-modules. Those specify dependencies based on a individual `setup.py` file per module.

## Project structure

The project includes 3 `pip` installable packages, namely:
- `so_tag_classifier_core`
- `so_tag_classifier_prediction`
- `so_tag_classifier_training`

`so_tag_classifier_core` includes data transformation methods that are used to transform input data for both training and prediction making. It also includes the set of tags we are interested in predicting. For the purpose of simplicity, the classifier is limited to the top 100 labels/tags as extracted on Nov 18, 2020 from a random sample of 100,000 StackOverflow posts. The rest are not predicted. This list can be modified directly in there.

`so_tag_classifier_prediction` is the package that can be used directly by a service to make predictions. It can be used for other types of problems as well, since the only limitation is that the input is a `string`. You can import `predict.predict` to make predictions from elsewhere, or you can run it as a script in the terminal, such as:

```bash
python3 predict.py -t "I would like to know how I can aggregate in MySQL efficiently" -mp ~/model.pkl
```

Note that the model path needs to be provided externally. Currently it needs to be a local path, but I will later add functionality to make it possible to load from S3.

`so_tag_classifier_training` is a package that can be used to run a training pipeline based on new data inputs. It builds a `scikit-learn` Pipeline and executes a `RandomizedSearchCV` for hyperparameter tuning. The set of space to explore can be configured in `params/parameter_tuning.py`.

You can install any of these packages by running:

```bash
cd stackoverflow-topic-classifier
pip install stackoverflow-topic-classifier/core
# pip install stackoverflow-topic-classifier/prediction
# pip install stackoverflow-topic-classifier/training
```

## Running the gRPC service

We've called the gRPC service `Nostradamus`, because it predicts the future :smile:, hopefully with better results that [the famous astrologer himself](https://en.wikipedia.org/wiki/Nostradamus). The basic way to run the service is the following:

1. Ensure all the local packages you need are installed

```bash
pip install stackoverflow-topic-classifier/core
pip install stackoverflow-topic-classifier/prediction
```

2. Ensure all gRPC dependencies are covered.

```bash
pip install grpcio
pip install grpcio-reflection
```

3. Run the server in one terminal (note that it is run by default on port 50051)

```bash
python nostradamus.service/nostradamus_server.py
```

4. And in another terminal, run:

```bash
python nostradamus.service/nostradamus_client.py --host localhost --port 50051
```

Note that the client implementation is just a demo to see how you would ask for a StackOverflow label prediction using the Python stub.
