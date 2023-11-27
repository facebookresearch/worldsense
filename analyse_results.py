"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os.path
from argparse import ArgumentParser
from worldsense.benchmark import analyse_results_in_testset_dir


parser = ArgumentParser()
parser.add_argument('-t', '--testset_dir', default="data/worldsense/test_set", type=str, required=False)
args = parser.parse_args()

testset_dir = args.testset_dir
analyse_results_in_testset_dir(testset_dir)

