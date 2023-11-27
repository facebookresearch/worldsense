"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os.path
import json
import random

from argparse import ArgumentParser
from tqdm import tqdm
from worldsense.benchmark import load_testset

# The following function correpsonds to calling a silly "language" model
# that doesn't even look at the question text, and answers one of the
# possible_responses at random.

def get_random_model_response(question_text, possible_responses):
    """Returns a response to the question_text, from the list 
    of possible_responses, according to the model.
 
    If no response was determined, the call can return an empty string "" 
    (or equivalently None).

    For a real language model this can happend if it failed to provide 
    a response that can be understood as onr of the possible_responses. 
    (even if queried a second time while insisting it must provide a 
    response among those and nothing else).
    """

    if random.random() < 0.005:  # Return a no-response in 0.5% of cases
        resp = ""    # means no-response
    else:  # choose resp randomly amongst possible reponses
        resp = random.choice(possible_responses)
    return resp


def evaluate_random_model_on_test_directory(testset_dir):
    print()
    print("*** EVALUATING RANDOM MODEL ON TEST SET", testset_dir) 

    results_dir = os.path.join(testset_dir, "results")

    prompting = "basic"
    modelname = "random"

    # It is essential that the file containing model responses follows this format
    # for its name, for it to be found by the results analysis code.
    results_jsonl = os.path.join(results_dir, f"{prompting}___{modelname}___results.jsonl")
    
    print()
    print("Loading test set:", testset_dir)
    trials_df = load_testset(testset_dir)
    print("It contains", trials_df.shape[0], " records, each with a question text for the language model to answer.")
    print()
    
    evaluate_random_model_on_trials(trials_df, results_jsonl)

    
def evaluate_random_model_on_trials(trials_df, results_jsonl):

    n = trials_df.shape[0]
    print("Writing responses of random model to file:", results_jsonl)

    with open(results_jsonl, 'w') as resultsfile:    
        for i in tqdm(range(n)):
            # print(f"Getting a response for question #{i+1} of {n}")
            Key = int(trials_df.at[i, 'Key'])     # a unique Key
            question_text = trials_df.at[i, 'text']   # the text to ask
            possible_responses = trials_df.at[i, 'expectedresp']   # list of possible responses

            # get model response
            resp = get_random_model_response(question_text, possible_responses)

            if resp is not None and resp != "" and resp not in possible_responses:
                raise RuntimeError('resp has to be one of the possible_responses or else None or the empty string ""')
            
            # append record to results resultsfile
            record = {'Key': Key, 'resp': resp}
            json.dump(record, resultsfile)
            resultsfile.write('\n')
            resultsfile.flush()

    print("Finished writing", n, "responses of random model to file:", results_jsonl)
    print()


    
## main

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--testset_dir', default="data/worldsense/test_set", type=str, required=False,
                        help="Path to the test set directory to apply the language model to.")
    args = parser.parse_args()
    
    testset_dir = args.testset_dir
    random.seed(713875139)
    evaluate_random_model_on_test_directory(testset_dir)

    print("Note: for analysing the recorded model's answers you can call:")
    print("  python analyse_results.py -t", testset_dir)
    print()

