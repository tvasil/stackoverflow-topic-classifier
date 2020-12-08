#!/usr/bin/env python

from setuptools import setup

install_requires = ["so-tag-classifier-core==1.0"]

setup(
    name="so-tag-classifier-prediction",
    version="1.0",
    description="The Prediction module for a multilabel classifier for StackOverflow post tags",
    author="Tania Vasilikioti",
    packages=["so_tag_classifier_prediction"],
    install_requires=install_requires,
)
