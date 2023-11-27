"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from setuptools import setup, find_packages

setup(
    name="world-sense-release",
    # use this if you want to directly import files inside worldsense
    # package_dir={"": "worldsense"},
    packages=find_packages(),
)