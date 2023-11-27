"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os.path
import pandas as pd
# from typing import Any, Dict, Tuple

from .dataloading import (
    load_trials_as_dataframe,
    find_files_ending_in,
    load_and_assemble_trials_and_results)
from .analysis import (
    add_fields_for_finegrained_analysis,
    compute_acc_table,
    accuracy_by_group,
    bias_by_group,
    pivot_pretty)


def load_testset(testset_dir = "data/worldsense/test_set",
                 add_extra_fields=False) -> pd.DataFrame:
    """Loads the test set located at the specified directory testset_dir
    and returns it as a pandas dataframe.

    If add_extra_fields is set to True, additional fields will be computed 
    as additional columns to the dataframe. These can be useful
    for finer-grained grouping/splitting/filtering of the data."""
    
    if not os.path.isdir(testset_dir):
        raise FileNotFoundError("No such testset diretory: " + testset_dir)
    trials_file = os.path.join(testset_dir, "trials.jsonl.bz2")
    trials_df = load_trials_as_dataframe(trials_file)
    if add_extra_fields:
        add_fields_for_finegrained_analysis(trials_df)
    return trials_df


def load_trainset(trainset_file = "data/worldsense/training_set/trials_100k.jsonl.bz2")  -> pd.DataFrame:
    """Loads a training set provied in .jsonl.bz2 format into memory as a pandas dataframe.
    Beware that the format of the returned dataframe is markedly different from that of testsets returned by load_testset. See README.md for explanation.
    """ 
    if not os.path.isfile(trainset_file):
        raise FileNotFoundError("No such training set file: " + trainset_file)        
    df = pd.read_json(trainset_file, orient='records', lines=True)
    return df


def analyse_results_in_testset_dir(testset_dir):
    """Analyses accuracy and bias of all model retults files present in the testset_dir's results/ subdirectory, and prints retults summary tables. 
    testset_dir must be the path to a directory containing a 'trials.jsonl.bz2' file and a 'results/' subdirectory.
    See README.md for a detailed description of the expected directory structure, file naming convention and formats."""

    print()
    print("**************************************************************************")
    print("*")
    print("*  ANALYSING RESULTS OF TESTSET", testset_dir) 
    print("*")
    print("**************************************************************************")
    print()
    testset_results_dir = os.path.join(testset_dir, "results")
    trials_file = os.path.join(testset_dir, "trials.jsonl.bz2")
    results_files = find_files_ending_in(testset_results_dir, "___results.jsonl")

    results_df = load_and_assemble_trials_and_results(trials_file, results_files)

    analyse_results_df(results_df)


def analyse_results_df(results_df):
    """Analyses accuracy and bias of all models results consolidated in an assembled pandas dataframe results_df, and prints results summary tables. 
    results_df must be a pandas dataframes that contains both trials fields and model response 'resp', as retuned by e.g. load_and_assemble_trials_and_results."""
    
    print("Computing accuracies and biases...")
    df_test_acc = compute_acc_table(results_df)

    # Additional fields useful for further grouping or filtering the data
    # for more finegrained analysis can be obtained by uncommenting this: 
    # add_fields_for_finegrained_analysis(results_df)
    # print("All fields:", results_df.keys())
    
    if len(df_test_acc['problemname'].unique()) > 1:
        print("")
        print("--------------------------------------------------------------------------")
        print("    AVERAGE ACCURACY ACROSS ALL PROBLEMS (with 95% confidence interval)")
        print("--------------------------------------------------------------------------")
        accst = accuracy_by_group(df_test_acc, ['prompting','modelname'])
        tab = pivot_pretty(accst, index=['modelname'], columns=['prompting'],
                           cell_style='str', scale=100.)
        print(tab)

    print()
    print("--------------------------------------------------------------------------")
    print("    ACCURACY for each of the problems  (with 95% confidence interval)")
    print("--------------------------------------------------------------------------")
    acc_st = accuracy_by_group(df_test_acc, ['prompting','modelname','problemname'])
    acc_tab = pivot_pretty(acc_st, index=['prompting','modelname'], columns=['problemname'],
                           cell_style='str', scale=100.)
    print(acc_tab)

    print()
    print("--------------------------------------------------------------------------")
    print("    BIAS for each of the problems (with 95% confidence interval)")
    print("--------------------------------------------------------------------------")
    bias_st = bias_by_group(df_test_acc, ['prompting','problemname','modelname'])
    bias_tab = pivot_pretty(bias_st, index=['prompting','modelname'], columns=['problemname'],
                            cell_style='str', scale=1.)
    print(bias_tab)

