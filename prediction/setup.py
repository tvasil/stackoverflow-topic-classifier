#!/usr/bin/env python

from setuptools import setup

install_requires = ["so-tag-classifier-core==1.0", "boto3>=1.16.33"]

setup(
    name="so-tag-classifier-prediction",
    version="1.0",
    description="The Prediction module for a multilabel classifier for StackOverflow post tags",
    author="Tania Vasilikioti",
    packages=["so_tag_classifier_prediction"],
    package_data={"so_tag_classifier_prediction": ["*.yml"]},
    install_requires=install_requires,
)
