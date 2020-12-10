#!/usr/bin/env python

from setuptools import setup

install_requires = ["joblib>=0.17.0", "nltk>=3.5", "scikit-learn==0.23.2", "pyyaml==5.3.1"]

extras_require = {"dev": ["pytest", "pre-commit"]}

setup(
    name="so-tag-classifier-core",
    version="1.0",
    description="The Core for a multilabel classifier for StackOverflow post tags",
    author="Tania Vasilikioti",
    packages=["so_tag_classifier_core"],
    install_requires=install_requires,
    extras_require=extras_require,
)
