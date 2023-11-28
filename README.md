# WorldSense 🌍

This is the repository for the data and analysis code of the *WorldSense benchmark* presented in the paper:

**[WorldSense: A Synthetic Benchmark for Grounded Reasoning
in Large Language Models](https://arxiv.org/abs/2311.15930)**, Youssef Benchekroun, Megi Dervishi, Mark Ibrahim
Jean-Baptiste Gaya, Xavier Martinet, Grégoire Mialon, Thomas Scialom,
Emmanuel Dupoux, Dieuwke Hupkes, Pascal Vincent. November 2023.


## Installation

1. `clone` repo
2. `cd` into repo
3. `pip install -r requirements.txt`

## Running basic result analysis for the baseline models from the paper

From inside the repo run:
```bash
python analyse_results.py -t data/worldsense/test_set
```

This should produce the following output:

```
**************************************************************************
*
*  ANALYSING RESULTS OF TESTSET data/worldsense/test_set
*
**************************************************************************

Loading trials data from data/worldsense/test_set/trials.jsonl.bz2
shape: (87048, 15)
  - Loading results file data/worldsense/test_set/results/basic___GPT3.5___results.jsonl
    shape: (87048, 4)
  - Loading results file data/worldsense/test_set/results/basic___GPT4___results.jsonl
    shape: (87048, 4)
  - Loading results file data/worldsense/test_set/results/basic___Llama2-chat___results.jsonl
    shape: (87048, 4)
  - Loading results file data/worldsense/test_set/results/basic___Llama2-FT+1M___results.jsonl
    shape: (87048, 4)
Assembling all into a large dataframe.
Computing accuracies and biases...

--------------------------------------------------------------------------
    AVERAGE ACCURACY ACROSS ALL PROBLEMS (with 95% confidence interval)
--------------------------------------------------------------------------
prompting          basic
modelname
GPT3.5        55.6 (0.4)
GPT4          75.6 (0.4)
Llama2-chat   56.2 (0.3)
Llama2-FT+1M  77.4 (0.4)

--------------------------------------------------------------------------
    ACCURACY for each of the problems  (with 95% confidence interval)
--------------------------------------------------------------------------
problemname            Infer.trivial Infer.normal Consist.trivial Consist.normal Compl.trivial Compl.normal
prompting modelname
basic     GPT3.5          64.8 (1.1)   55.1 (0.6)      52.2 (0.7)     49.9 (0.7)    59.9 (1.0)   51.8 (1.0)
          GPT4            90.3 (0.7)   75.6 (0.6)      71.2 (0.6)     65.0 (0.6)    93.1 (0.6)   58.5 (0.7)
          Llama2-chat     62.2 (0.8)   59.1 (0.5)      54.2 (0.4)     49.4 (0.3)    60.5 (0.7)   51.5 (0.5)
          Llama2-FT+1M    79.7 (0.9)   80.6 (0.5)      54.9 (0.3)     52.8 (0.3)    98.5 (0.2)   97.8 (0.3)

--------------------------------------------------------------------------
    BIAS for each of the problems (with 95% confidence interval)
--------------------------------------------------------------------------
problemname            Infer.trivial  Infer.normal Consist.trivial Consist.normal Compl.trivial Compl.normal
prompting modelname
basic     GPT3.5         0.08 (0.02)  -0.34 (0.01)    -0.15 (0.01)   -0.00 (0.01)  -0.02 (0.02)  0.20 (0.02)
          GPT4          -0.14 (0.01)  -0.20 (0.01)    -0.12 (0.01)    0.00 (0.01)   0.09 (0.01)  0.78 (0.01)
          Llama2-chat   -0.43 (0.02)  -0.63 (0.01)     0.52 (0.01)    0.79 (0.01)   0.52 (0.02)  0.83 (0.01)
          Llama2-FT+1M   0.26 (0.02)   0.04 (0.01)     0.84 (0.01)    0.90 (0.01)  -0.02 (0.00)  0.02 (0.01)
```


## Test sets

The repo contains several test sets, all located inside `data/worldsense/`. 

Name | Testset directory | Description | Number of records (lines)
--- | --- | --- | ---
**test_set** | data/worldsense/test_set | The official WorldSense benchmark test set  | 87048
**memorisation** | data/worldsense/other_tests/memorisation | To test to what degree fine-tuned models memorized their training data  | 11232
**ood-size**   | data/worldsense/other_tests/ood-size | standard problems but with size 6 | 9360
**ood-query** | data/worldsense/other_tests/ood-query | additional problem named *Infer.extrema* to probe first, last relation| 6840
**ood-problem** | data/worldsense/other_tests/ood-problem | additonal problem named *Infer.parallel* requiring 2-dimensional reasoning| 5400


Each *record* contains a question `text`, that can be given to a LLM to answer.
The official **test_set** contains questions for the 6 standard problem variants desribed in the paper, with lengths 3, 4 and 5.

The `analyse_results` script can analyse the results included for any of these test sets with its option `-t` (which defaults to the official test set) E.g.:
```bash
python analyse_results.py -t data/worldsense/other_tests/ood-size
```

## Format and loading of test sets

Each test-set directory listed above contains a file named `trials.jsonl.bz2`
This file is a bzip2-compressed **.jsonl** file in [JSON-Lines format](https://jsonlines.org).
Each line contains a *record* as a dictionary mapping fieldnames to values. These records will also be called *trials*.

It can easily be read in as a pandas dataframe as follows:

```python
from worldsense.benchmark import load_testset

testset_dir = "data/worldsense/test_set"
trials_df = load_testset(testset_dir)  # Load all trials as a pandas dataframe
```

The `load_testset` function under the hood calls `pandas.read_json('trials.jsonl.bz2', orient='records', lines=True)` and does minimal cleaning (s.a. deobfuscating the `goldresp` field).

What one then gets is a pandas dataframe with several column fields, the most important being:
- `Key`: a unique integer identifier for the record
- `text`: the question text to ask a language model
- `expectedresp`: the list of acceptable responses
- `goldresp`: the correct response
- `problemname`: the problem that the question is an instance of
- `skin`: the skin that was used for rendering the question

Additional fields are there and allow for more or less fine grained analysis of accuracy results.


## Model results files

### The results subdirectory

Each test-set directory also contains, besides the `trials.jsonl.bz2` file, a `results/` subdirectory. It contains one result file for each language model that was tested on that test set. E.g.:

```bash
ls data/worldsense/test_set/results
```
will show:
```
basic___GPT3.5___results.jsonl
basic___GPT4___results.jsonl
basic___Llama2-FT+1M___results.jsonl
basic___Llama2-chat___results.jsonl
```

### Naming scheme for results files

Result files follow a standard naming scheme:
`<prompting>___<modelname>___results.jsonl`

- *modelname* should indicate a specific language model and version (e.g. GPT4) and should also indicate if/how it has been finetuned.
- *prompting* should indicate the prompting strategy employed. It should be set to `basic` to indicate that the question text was asked as is. Otherwise if experimenting with different prompting strategies (i.e. modifying or complementing the basic question-text s.a. inserting chain-of-thought instructions, or providing few-shot examples as additional context) it sould be used to specify what prompting strategy was employed.

**Note:** In the filename it is important to respect the *triple* underscores, and the ending in `___results.jsonl` so that the `analyse_results.py` script can find these files and parse their name.

*modelname* and *prompting* are only used in reporting (for grouping results). Experimenters can set them to whatever they wish, to indicate their model and prompting strategy.


### Format of results files

Result files are (uncompressed) **.jsonl** files in [JSON-Lines format](https://jsonlines.org), with one dictionary record per line. These records are very simple as illustrated here:

```bash
head -10 data/worldsense/test_set/results/basic___GPT4___results.jsonl
```
will show:
```
{"Key":-276741083417243227,"resp":"1"}
{"Key":2747235547611487721,"resp":"1"}
{"Key":2917042815647934077,"resp":"3"}
{"Key":4803862128065633392,"resp":"1"}
{"Key":-2266038382228944547,"resp":"1"}
{"Key":-8972465100086251897,"resp":"3"}
{"Key":3540930086427752797,"resp":"TRUE"}
{"Key":633310297737009100,"resp":"FALSE"}
{"Key":6223161607606023176,"resp":"IMPOSSIBLE"}
{"Key":-5701584567800374128,"resp":"IMPOSSIBLE"}
```

Each record has only 2 fields:
- `Key` matches the `Key` in the dataset's trials (the dataframe returned by `load_testset` above).
- `resp` is the response given by the model to the corresponding trials' question `text`. It must be one of the possible responses from the `expectedresp` list for that question. Or else if the model failed to provide one of these (even after being insistingly requeried),`resp` should be the empty string "".


## How to test another language model or prompting strategy

Testing another model is thus a simple matter of:
1. loading one of the test sets with `load_testset` as shown above.
2. open a results file following the naming scheme `<prompting>___<modelname>___results.jsonl` for writing inside that test-set's `results/` subdirectory. *modelname* should indicate your model's name and version and *prompting* should indicate your prompting strategy (use `basic` if asking the question diretly). Make sure you use *triple underscores* in the filename to separate these.
3. looping over the test-set records, and for each:
   - ask the model the question `text`
   - extract the model's response
   - verify that the model's response matches one of the acceptable responses in `expectedresp`. If not try reprompting the model e.g. with `Only respond with one of these options:[expectedresp]`. If the answer is still not a valid one set the response to the empty string ""
   - append this reponse as `resp` together with the associated `Key` to the results file.
4. To get Worldsense accuracy score and basic analysis of the results, simply run `python analyse_results.py -t` *testset-directory* 

This procedure is implemented in the simple `test_random_model.py` script. It employs a silly "language model" that just randomly picks an answer amongst the allowed `expectedresp`. Thus if you run
```bash
python test_random_model.py -t data/worldsense/test_set
```
it will create a new results file called `basic___random___results.jsonl` inside `data/worldsense/test_set/results/`. That file will contain the responses of the random model.

If you then re-run
```bash
python analyse_results.py -t data/worldsense/test_set
```
the reported accuracy and bias tables will now have a new row for model `random` (under *prompting*=`basic`).

Remark: Expected chance level *accuracy* is 50% for all problems. Expected *bias* is 0 for this random LLM for both *inference* and *consistency* problems. But for the *completeness* problems (which have 3 possible responses) it is expected to be around 0.33, not 0. Since that model uniformly samples one of the 3 acceptable responses, it will 2/3 of the times pretend it knows the answer and only 1/3 of the time say it is not possible to decide. For this problem, *bias* mesures KNOWN (folding response 1 and 2) v.s. UNKNOWN (response 3), thus the random model will predominantly pretend it knows the answer rather than saying it's not decidable, hence the positive bias.

**Notes:**
   - The code of `test_random_model.py` can easily be adapted to query your real language model. 
   - The simple results file format makes it easily amendable to pausing/interrupting and resuming appending results to it.
   - `analyse_results.py` will also happily analyse *partial* results files: it outputs average accuracy estimates even if not all the results are in yet (or if you don't want to run a costly language model on all 87048 rows of the official test set). In this case you'll obtain larger confidence intervals, that should shrink as more results are appended to the file.

## Training sets (to fine tune LLMs)

Training sets are also provided for fine-tuning LLMs on the set of standard problems we test. These training sets use different "skins" than the standard test set.

Name | Training-set file | Number of records (lines)
--- | --- | ---
trials_10k | data/worldsense/training_set/trials_10k.bz2 | 12096
trials_100k | data/worldsense/training_set/trials_100k.bz2 | 108864
trials_1M | Downloadable 38Mb file <http://dl.fbaipublicfiles.com/worldsense/trials_1M.jsonl.bz2> | 1091664

**Important note:** These training set files are distributed in bzip2-compressed [JSON-Lines format](https://jsonlines.org), with one dictionary record per line. But **the format of each record is quite different form that of test set files**. Instead the file conforms to a typical format used for fine-tuning Llama-2 chat LLMs.

Each record is made of 2 fields:

- `dialog_history`: dialogue as an array of messages each containg a role and content 
- `target_message`: specifies target. For example: `POSSIBLE`, `IMPOSSIBLE`, `TRUE`, `FALSE`, etc.

Example of such a record (pretty printed json):

```json
{
"dialog_history":
  {
  "messages":
    [
      {
      "role":"user",
      "content":"Over his lifetime, Grandpa planted 3 trees in his garden... \nOnly respond with one of these 2 options: 'TRUE', 'FALSE' without any explanation."
      }
    ]
  },
"target_message":"TRUE"
}
```

The training sets can be loaded in memory via e.g.:

```python
from worldsense.benchmark import load_trainset

trainset = load_trainset("data/worldsense/training_set/trials_100k.jsonl.bz2")
```


## Remarks on analysing results, in case you want to do a more fine-grained analysis or fancier display of statistics

The basic analysis printed by the `analyse_results.py` script is done by calling functions defined in `worldsense/benchmark.py`:
- `analyse_results_in_testset_dir` loads and assembles (joins) the test set's trials file and the associated results files into a single large pandas dataframe. It then calls `analyse_results_df`.
- `analyse_results_df` computes the basic errors and statistics and prints them.

`analyse_results_df` is the function you should take as starting inspiration if you want to do display or plot result statistics differently (e.g. within a jupyter notebook) or do a finer grained analysis (by filtering and grouping the data differently). For this it is important to follow similar steps to what `analyse_resuts_df` does, namely always:
1. Call `compute_acc_table`: this will reduce each dependent (non i.i.d.) tuple of trials to a single row and compute the appropriately weighted accuracy (`acc` field) and `bias`.
2. Call either `accuracy_by_group` to compute a table containing the average accuracy within each group or `bias_by_group` to get the corresponding bias.
It is important to call these functions rather than doing a normal `groupby` (or `pivot`) aggregation by yourself: they do a proper equal reweighting before aggregating across `problemsize` and before aggregating across `problemname`. They also correctly compute the corresponding confidence intervals.
3. [Optionally] call `pivot_pretty` to get a display-friendly pivoted version of the accuracy or bias table obtained in step 2.  `pivot_pretty` allows to choose what to put in rows (index) and columns, and to have each cell contain a nicely formatted string that combines both value and confidence interval (alternative content formatting for each cell is available by specifying a different `cell_style` argument, see documentation of `format_cell` function in `worldsense/analysis.py` for possibilities).

# Citation

```
@article{benchekroun2023worldsense,
      title={WorldSense: A Synthetic Benchmark for Grounded Reasoning in Large Language Models}, 
      author={Youssef Benchekroun and Megi Dervishi and Mark Ibrahim and Jean-Baptiste Gaya and Xavier Martinet and Grégoire Mialon and Thomas Scialom and Emmanuel Dupoux and Dieuwke Hupkes and Pascal Vincent},
      year={2023},
      eprint={2311.15930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

