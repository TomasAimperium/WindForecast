#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path
from setuptools import find_packages, setup


# Where the magic happens:
setup(
    name="windspeed",
    version="0.0.1",
    description="WindPredictPackage",
    long_description="file: README.md",
    long_description_content_type='text/markdown',
    author="Tomas",
    author_email="tomas.alcaniz@aimperium.com",
    python_requires=">=3.6",
    url='your github project',
    packages=find_packages(exclude=('tests',)),
    package_data={'windspeed': ['VERSION']},
    install_requires="",
    extras_require={},
    include_package_data=True,
    license='LICENSE',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
