#!/usr/bin/env python

import os

from setuptools import setup

install_requires = ["numpy>=1.19.2", "so-tag-classifier-core==1.0", "boto3>=1.16.33"]

setup(
    name="so-tag-classifier-training",
    version="1.0",
    description="The Training module for a multilabel classifier for StackOverflow post tags",
    author="Tania Vasilikioti",
    packages=["so_tag_classifier_training"],
    package_data={"so_tag_classifier_training": ["*.yml"]},
    install_requires=install_requires,
)
