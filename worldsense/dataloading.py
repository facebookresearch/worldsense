"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pandas as pd
import os
from typing import List, Tuple
import pathlib
import pickle
import json
from datetime import datetime


#########################################################
# FILE UTILITY AND DATA-LOADING FUNCTIONS

def verboserename(src, dest, verbose=True):
    if verbose:
        print("RENAMING FILE", src, "->", dest)
    os.rename(src, dest)

    
def verboseremove(filepath, verbose=True):
    if verbose:
        print("REMOVING FILE", filepath)
    os.remove(filepath)


def find_files_ending_in(directory_path: str, ending: str):
    """Returns a list all paths to files ending in ending, 
    searching recursively inside directory_path"""
    filepaths = []
    for root, directories, files in os.walk(directory_path):
        for file in files:
            if file.endswith(ending):            
                file_path = os.path.join(root, file)
                filepaths.append(file_path)
    return filepaths

    
def make_timestamp_file(directory, basename="CREATED"):
    filename = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss") + "__" + basename
    filepath = os.path.join(directory, filename)
    f = open(filepath, "w")
    f.write(filepath + "\n")
    f.close()

    
def load_pkls_file_as_list(pkls_file):
    """Loads a file made of a *sequence of* pickle serializations. Returns
    these in a list.  This is designed to load all valid records, even if a
    crash made the last record incomplete and thus corrupedand invalid.
    """
    records = []
    with open(pkls_file, 'rb') as f:
        while True:
            try:
                record = pickle.load(f)
            except (EOFError, IOError, pickle.PickleError) as error:
                break
            records.append(record)
    return records


def infer_expname_and_expdir(filepath: str) -> Tuple[str, str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} : no such file or directory.")
    
    if (filepath.endswith("trials.json")
        or filepath.endswith("trials.jsonl")
        or filepath.endswith("trials.json.bz2")
        or filepath.endswith("trials.jsonl.bz2")):        
        expdir_path = pathlib.PurePath(filepath).parent
    elif (filepath.endswith("__allresults.pkls")
          or filepath.endswith("__allresults.pkls.partial")
          or filepath.endswith("___results.jsonl")):
        expdir_path = pathlib.PurePath(filepath).parent.parent
    elif (os.path.exists(os.path.join(filepath, "trials.json"))
          or os.path.exists(os.path.join(filepath, "trials.jsonl"))
          or os.path.exists(os.path.join(filepath, "trials.json.bz2"))
          or os.path.exists(os.path.join(filepath, "trials.jsonl.bz2"))):
        expdir_path = pathlib.PurePath(filepath)
    else: # fallback
        expdir_path = pathlib.PurePath(filepath).parent.parent
        
    if not is_experiment_directory(expdir_path):
        raise FileNotFoundError(f"Could not determine experiment direcotry associated with {filepath}")
    
    expname = expdir_path.parts[-1]
    return expname, str(expdir_path)


def infer_modelname_and_prompting(filepath):
    """returns modelname, prompting inferred from filenames of the form:
            'prompting___modelname___results.jsonl'
            or 'modelname__allresults.pkls'
            or 'modelname__allresults.pkls.partial'
    In the latter two cases, since there is no prompting in the filename,
    the call will return modelname,prompting with prompting=None.
    """
    filename = os.path.basename(filepath)
    if (filename.endswith('__allresults.pkls')
        or filename.endswith('__allresults.pkls.partial')):
        modelname = filename.split('__')[0]
        prompting = None
    elif filename.endswith('___results.jsonl'):
        prompting, modelname, _ = filename.split('___')
    else:
        raise ValueError(
            """Can only infer modelname and prompting from filenames of the form
            'prompting___modelname___results.jsonl'
            or 'modelname__allresults.pkls'
            or 'modelname__allresults.pkls.partial'
            """)
    return modelname, prompting

            
def load_experiment_specs(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    # Initialize an Experiment with data from the JSON file
    else:
        raise ValueError(f"Incorrect filename '{filename}', file not found.")
    experimentname = data["experimentname"]
    summary = data["summary"]
    params = data["params"]
    seed = data["seed"]
    #    expe=Experiment(experimentname,summary,params)
    return experimentname, summary, seed, params


def endswith(s, endings=[]):
    if not isinstance(endings, list):
        endings = [endings]
    return any([s.endswith(ending) for ending in endings])
    

def read_json_as_dataframe(json_filepath):
    VALID_JSON_ENDINGS = ['.json',
                          '.json.zip',
                          '.json.gz',
                          '.json.bz2']
    VALID_JSONL_ENDINGS = ['.jsonl',
                           '.jsonl.zip',
                           '.jsonl.gz',
                           '.jsonl.bz2']
    if endswith(json_filepath, VALID_JSONL_ENDINGS):
        return pd.read_json(json_filepath, orient='records', lines=True)
    elif endswith(json_filepath, VALID_JSON_ENDINGS):
        return pd.read_json(json_filepath)
    raise ValueError(f"Unsupported extension for read_json_as_dataframe: {json_filepath} Filename should end in one of {VALID_JSON_ENDINGS + VALID_JSONL_ENDINGS}")


GOLDRESP_OBFUSCATION = [
    ("TRUE", "Emmanuel"),
    ("FALSE", "Megi"),
    ("POSSIBLE", "Dieuwke"),
    ("IMPOSSIBLE", "Pascal"),
    ("1", "Mark"),
    ("2", "Youssef"),
    ("3", "Yoda")]


def obfuscate_goldresp_inplace(df):
    """
    Replaces goldresp field by obfuscated goldresp_obfcus field 
    where replacements are given by GOLDRESP_OBFUSCATION
    """
    if 'goldresp' not in df:
        raise ValueError("Dataframe does not have a goldresp field")

    df['goldresp'] = df['goldresp'].replace(
        to_replace = [a for a,b in GOLDRESP_OBFUSCATION],
        value = [b for a,b in GOLDRESP_OBFUSCATION])
    df.rename(columns={'goldresp': 'goldresp_obfusc'}, inplace=True)

    
def deobfuscate_goldresp_inplace(df):
    """
    Replaces goldresp_obfusc field by a deobfuscated goldresp field 
    where replacements are given by GOLDRESP_OBFUSCATION
    """
    if 'goldresp_obfusc' not in df:
        raise ValueError("Dataframe does not have a goldresp_obfusc field")

    df.rename(columns={'goldresp_obfusc': 'goldresp'}, inplace=True)
    df['goldresp'] = df['goldresp'].replace(
        to_replace = [b for a,b in GOLDRESP_OBFUSCATION],
        value = [a for a,b in GOLDRESP_OBFUSCATION])

    
def load_trials_as_dataframe(
        trials_json_path,
        include_expname_fields=True,
        include_params_fields=False,
        auto_deobfuscate=True):
    if not os.path.exists(trials_json_path):
        raise FileNotFoundError("File " + trials_json_path + " does not exist")
    df = read_json_as_dataframe(trials_json_path)
    if include_expname_fields:
        expname, expdir = infer_expname_and_expdir(trials_json_path)
        df['EXPNAME'] = expname
        df['EXP_i'] = range(df.shape[0])
    if auto_deobfuscate:
        if 'goldresp_obfusc' in df:
            deobfuscate_goldresp_inplace(df)
    if include_params_fields:
        expname, expdir = infer_expname_and_expdir(trials_json_path)
        spec_file = os.path.join(expdir, expname + ".json")
        if os.path.exists(spec_file):
            experimentname, summary, seed, params = load_experiment_specs(spec_file)
            if isinstance(params, list):
                params = params[0]
                for key, val in params.items():
                    df["PARAM_" + key] = repr(val)
    return df

        
def load_results_as_dataframe(filepath, add_modelname_and_prompting=True):
    if filepath.endswith('.jsonl'):
        df = pd.read_json(filepath, orient='records', lines=True)
    else:
        raise ValueError("only supporting .jsonl files")
    if add_modelname_and_prompting:
        modelname, prompting = infer_modelname_and_prompting(filepath)
        df['modelname'] = modelname
        df['prompting'] = prompting
    return df


def load_and_assemble_trials_and_results(trials_file, results_files, expname=None, verbose=True):
    if verbose:
        print("Loading trials data from", trials_file)
    trials_df = load_trials_as_dataframe(trials_file)
    n = trials_df.shape[0]
    if verbose:
        print("shape:", trials_df.shape)
    results_df_list = []
    for filepath in results_files:
        if verbose:
            print("  - Loading results file", filepath)
        results_df = load_results_as_dataframe(filepath)
        if verbose:
            print("    shape:", results_df.shape)
            rn = results_df.shape[0]
            if rn != n:
                print(f"    *** !!! [WARNING] PARTIAL RESULTS FILE: number of rows {rn} differs from {n}") 
        results_df_list.append(results_df)
    if verbose:
        print("Assembling all into a large dataframe.")
    full_df = assemble_trials_and_results_df(trials_df, results_df_list)
    return full_df


def assemble_trials_and_results_df(trials_df, results_df_list):
    """
    returns a single full_df assembled from a trials_df and list of results_df
    results_df_list is a list of dataframes containing fields:
      'modelname', 'prompting', 'Key', 'resp'
    """

    trials_df = trials_df.set_index('Key')
    joined_dfs = []
    for results_df in results_df_list:
        joined_df = pd.merge(trials_df, results_df, on='Key', how='inner')
        joined_dfs.append(joined_df)

    full_df = pd.concat(joined_dfs, axis=0)
    return full_df
                          


def load_experiment_results_as_dataframe(results_pkls_path,
                                         include_expname_fields=True,
                                         include_params_fields=False,
                                         trials_path=None):
    if trials_path is None:
        # infer trials_path from results_pkls_path
        expname, expdir = infer_expname_and_expdir(results_pkls_path)
        trials_path = os.path.join(expdir, "trials.json")
    # print("Loading trials")
    trials_df = load_trials_as_dataframe(
        trials_path,
        include_expname_fields=include_expname_fields,
        include_params_fields=include_params_fields)
    
    # print("Loading results list")
    results_list = load_pkls_file_as_list(results_pkls_path)
    res_filename = pathlib.Path(results_pkls_path).stem
    modelname = res_filename.split('__')[0]
    results_dict_list = []
    # print("Converting results list into dict list")
    RESPONSEFIELDS = ['fullresp1','fullresp2','resp','confidence']
    for text, responses in results_list:
        di = dict(zip(RESPONSEFIELDS, responses))
        di["text2"] = text
        results_dict_list.append(di)
    # print("Instantiating results_df")
    results_df = pd.DataFrame.from_dict(results_dict_list)

    trials_n = len(trials_df)
    results_n = len(results_df.index)
    if results_n < trials_n:
        # print("Making smaller copy")
        trials_df = trials_df.iloc[0:results_n].copy()

    # print("Adding modelname")
    trials_df["modelname"] = modelname

    # adding new field 'clarified' and 'context_nb' if not there
    if "clarified" not in trials_df:
        trials_df["clarified"] = 0
    if "context_nb" not in trials_df:
        trials_df["context_nb"] = 0

    # print("Concatenating")
    concat_df = pd.concat([trials_df, results_df], axis=1)

    if not concat_df["text"].equals(concat_df["text2"]):
        raise RuntimeError("Bug: text does not match between trials and results. This should never happen!")
    # print("Dropping text2")
    concat_df = concat_df.drop(columns=["text2"])

    return concat_df

        
def load_all_results(directory_list: List[str],
                     include_partials=False,
                     verbose=True):
    filepaths = []
    endings = ["__allresults.pkls"]
    if include_partials:
        endings.append("__allresults.pkls.partial")
        
    for ending in endings:
        for directory in directory_list:
            if directory.endswith(ending):
                filepaths.append(directory)
            else:
                filelist = find_files_ending_in(directory, ending)
                filepaths.extend(filelist)
    dfs = []
    for fpath in filepaths:
        if verbose:
            print("Loading results dataframe based on file", fpath)
        df = load_as_dataframe(fpath)
        dfs.append(df)

    # Concatenate all dataframes into a single large dataframe
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df
    

def load_all_trials(directory_list: List[str],
                    verbose=True):
    dfs = []
    for dirpath in directory_list:
        trials_file_path = get_trials_file(dirpath)
        if verbose:
            print("Loading trials dataframe from", trials_file_path)
        df = load_trials_as_dataframe(trials_file_path,
                                      include_expname_fields=True,
                                      include_params_fields=False)
        dfs.append(df)
    
    # Concatenate all dataframes into a single large dataframe
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df


def is_experiment_directory(path):
    return (os.path.isdir(path) and
            (os.path.isfile(os.path.join(path,    "trials.json"))
             or os.path.isfile(os.path.join(path, "trials.jsonl"))
             or os.path.isfile(os.path.join(path, "trials.json.bz2"))
             or os.path.isfile(os.path.join(path, "trials.jsonl.bz2"))
            ))


def get_trials_file(directory):
    POSSIBLE_FNAMES = [
        "trials.json",
        "trials.jsonl",
        "trials.json.bz2",
        "trials.jsonl.bz2"]
    for fname in POSSIBLE_FNAMES:        
        fpath = os.path.join(directory, fname)
        if os.path.exists(fpath):
            return fpath
        

def experiment_exists(name: str, dir: str) -> bool:
    """Checks if directory exists"""
    parent_dir = os.path.join(dir, name)
    if os.path.exists(parent_dir):
        return True
    return False


def load_as_dataframe(filepath,
                      include_expname_fields=True,
                      include_params_fields=False):
    """
    Loads a dataframe of trials or trials with results.
    filepath must be either "trials.json"
    or must end in "__allresults.pkls"
    or in "__allresults.pkls.partial"
    It can also be an experiment directory, 
    in which case that directory's trials.json will be loaded and returned.
    """            
    df = None
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file or directory: {filepath}")
    if is_experiment_directory(filepath):
        df = load_as_dataframe(
            get_trials_file(filepath),
            include_expname_fields = include_expname_fields,
            include_params_fields = include_params_fields)
    elif os.path.basename(filepath).startswith("trials."):
        df = load_trials_as_dataframe(
            filepath,
            include_expname_fields=include_expname_fields,
            include_params_fields=include_params_fields)
    elif (filepath.endswith("__allresults.pkls")
          or filepath.endswith("__allresults.pkls.partial")):
        df = load_experiment_results_as_dataframe(
            filepath,
            include_expname_fields = include_expname_fields,
            include_params_fields = include_params_fields)
    else:
        raise ValueError(f"""Unrecognized file: {filepath} 
        Should be either "trials.json" (or .jsonl .jsonl.bz2, ...) 
        or an experiment directory containing a "trials.json"
        or a file ending in either "__allresults.pkls"
        or "__allresults.pkls.partial""")

    return df

