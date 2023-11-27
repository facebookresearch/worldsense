"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import pandas as pd

# Define order preferences to be used in pivot_pretty

ORDER_PREFS = {
    'problemname': [
        'trivial-ulist-complete-TF',
        'Infer.trivial',
        'complete-TF',
        'Infer.normal',
        'trivial-complete-PI',
        'Consist.trivial',
        'complete-PI',
        'Consist.normal',
        'incomplete-easy',
        'Compl.trivial',
        'incomplete-hard',
        'Compl.normal',
        'complete-unary-TF',
        'Infer.extrema',
        'complete-TF-parallel',
        'Infer.parallel',
    ],
    'modelname': [
        'gpt3.5',
        'GPT3.5',
        'gpt-4',
        'GPT4',
        'llama',
        'llama2_70b',
        'Llama2-chat',
        'llama2_70b_finetuned_0',
        'llama2_70b_finetuned_100k'
        'llama2_70b_finetuned_1M'
        'Llama2-FT+0',
        'Llama2-FT+100k',
        'Llama2-FT+1M',
        'llama_finetuned_0',
        'llama_finetuned_2k',
	'llama_finetuned_10k',
	'llama_finetuned_100k',
	'llama_finetuned_1M'],
    'prompting': [
        'basic',
        'direct',
        'clarified',
        'COT',
        'incontext_IID',
        'incontext_DIFFERENTCAT']
}


def datatype_counts(df):
    """Returns a dataframe with counts for the types of each field.
    columns 'dtype' gives the pandas dtype
    column isna reports the number of nans
    other columns correspond to actual python types and provide their counts."""
 
    result = pd.DataFrame({'dtype': df.dtypes})
    for fieldname in df.keys():
        result.loc[fieldname, 'size'] = df[fieldname].size
        result.loc[fieldname, 'count'] = df[fieldname].count()
        result.loc[fieldname, 'isna'] = df[fieldname].isna().sum()
        field_types = df[fieldname].transform(lambda x: type(x))
        type_counts = field_types.value_counts()
        for typeid, count in type_counts.iteritems():
            typestr = str(typeid)
            typestr = typestr.replace("<class '", "").replace("'>", "")
            result.loc[fieldname, typestr] = count
        # summary = str(field_types.unique()).replace("<class '", "'").replace("'>", "'")
        # result.loc[fieldname, 'python_types'] = summary
    result = result.fillna("")
    return result


def compare_dataframes(df1, df2):
    """
    Provides details on similarities and differences between two data frames.
    """
    print("** Comparing dataframes 1 and 2 **")
    print("SHAPE 1:", df1.shape)
    print("SHAPE 2:", df2.shape)
    print("")
    fields1 = df1.keys().tolist()
    fields2 = df2.keys().tolist()
    print("FIELDS 1:", fields1)
    print("FIELDS 2:", fields2)

    # print()
    # print("-----------------")
    # print("FIELDS 1 DATATYPES")
    # print(datatype_counts(df1))
    # print("")
    # print("-----------------")
    # print("FIELDS 2 DATATYPES")
    # print(datatype_counts(df2))
    # print("")
    # print("-----------------")
    # print("")
    
    common_fields = set(fields1) & set(fields2)
    print("COMMON FIELDS:", common_fields)
    fields_exclusive_to_1 = set(fields1).difference(set(fields2))
    print("FIELDS EXCLUSIVE TO 1:", fields_exclusive_to_1)
    fields_exclusive_to_2 = set(fields2).difference(set(fields1))
    print("FIELDS EXCLUSIVE TO 2:", fields_exclusive_to_2)
    
    n = min(df1.shape[0], df2.shape[0])
    df1 = df1.iloc[0:n]
    df2 = df2.iloc[0:n]
    
    matching = [fi for fi in common_fields if df1[fi].equals(df2[fi])]
    differing = [fi for fi in common_fields if not df1[fi].equals(df2[fi])]

    print("FIELDS WITH SAME VALUES:", matching)
    print("FIELDS WITH DIFFERING VALUES:", differing)
    #for fieldname in ["text", "goldresp", "resp"]:
    #    if fieldname in df1.keys() and fieldname in df2.keys():
    #        print("MATCHING", "'"+fieldname+"'", ":", df1["text"].equals(df2["text"]))


def reorder_categoricals_inplace(df, order_prefs=ORDER_PREFS):
    """Makes the fields in order_prefs categorical with the given preferred order
    for their possible values.
    This will affect in which order they will be displayed"""
    for key in df.keys():
        if key in order_prefs:
            order = order_prefs[key]
            uniques = df[key].unique()
            uniques_known = [v for v in order if v in uniques]
            uniques_unknown = [v for v in uniques if v not in uniques_known]
            uniques_reordered = uniques_known + uniques_unknown
            df[key] = df[key].astype("category").cat.reorder_categories(uniques_reordered)


def shuffle_df(df, keep_groups=None, seed=432832):
    """Returns a dataframe with the rows shuffled.
    If keep_groups is None it shuffles all rows randomly
    If keep_groups specifies a fieldname or a list of fieldnames
    then the shuffling wil shuffle these groups but keep together elements of each group
    """
    if keep_groups is None:
        # shuffle the entire dataframe
        df = df.sample(frac = 1, random_state=seed)
    else:
        groups = [df for _, df in df.groupby(keep_groups)]
        saved_state = random.getstate()
        random.seed(seed)
        random.shuffle(groups)
        random.setstate(saved_state)
        df = pd.concat(groups).reset_index(drop=True)
        
        # ids = df[keep_together_field].unique()
        # random.shuffle(ids)
        # df = df.set_index(keep_together_field).loc[ids].reset_index()
        
    return df

    
def compute_acc_table(df, filter_incomplete_tuples=True):
    """
    Adds the 'acc' field for accuracy and 'bias' for True/False bias.
    Will reduce the table to 1 row per tuple and
    remove some fields.
    Returns the new table.
    """

    df = df.copy()
    initial_n = df.shape[0]

    # must summarize tuples by one row

    # List of columns that will make no more sense once tuple has been summarized
    columns_to_remove = ['resp','goldresp','text']

    # First filter out incomplete tuples
    if filter_incomplete_tuples:
        df['tuplelen'] = df.groupby(['modelname','tuple_ID'])['resp'].transform("size")
        df['expectedresp_len'] = df['expectedresp'].apply(lambda x: len(x))
        df = df[df['tuplelen']==df['expectedresp_len']]
        if df.shape[0] != initial_n:
            print("!!! WARNING you have incomplete tuples in that dataframe !!! (this is normal if you chose to load partial results)")

    resp_map = {
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        "POSSIBLE": "POSSIBLE",
        "IMPOSSIBLE": "IMPOSSIBLE",
        "1": "KNOW",
        "2": "KNOW",
        "3": "UNKNOWN",
    }

    bias_map = {
        "TRUE": 1.,
        "FALSE": -1.,
        "POSSIBLE": 1.,
        "IMPOSSIBLE": -1.,
        "1": 1.,
        "2": 1.,
        "3": -1.,
        }

    weight_map = {
        "TRUE": 0.5,
        "FALSE": 0.5,
        "POSSIBLE": 0.5,
        "IMPOSSIBLE": 0.5,
        "1": 0.25,
        "2": 0.25,
        "3": 0.5,
    }
    # add weights for the weighted mean
    df['acc_weight'] = df['goldresp'].map(weight_map)
    # df['goldbias'] = df['goldresp'].map(bias_map)
    df['bias'] = df['resp'].map(bias_map)
    df['acc'] = (df['resp'].map(resp_map)==df['goldresp'].map(resp_map)).astype(int)

    # Compute a weighted mean
    df = transform_into_weighted_mean(df, ['modelname','tuple_ID'], ['acc','bias'], 'acc_weight')
    columns_to_remove.append('acc_weight')

    # Drop columns that make no more sense
    df.drop(inplace=True, columns=columns_to_remove)

    # Reduce each tuple to 1 row
    df = df.groupby(['modelname','tuple_ID']).head(1)

    return df


def bias_by_group(df,
                  by=['modelname','prompting','problemname','problemsize','complexity','bintern'],
                  rebalance=True):
    """Returns a dataframe containing mean,var,std,sem,count and conf95 of the 'bias' within each group.
    With rebalance=False this is the direct result of a mere groupby.aggregate
    With rebalance=True the call does equal rebalancing of problemsize (if aggregating across problemsizes)
    Also if aggregating on problemname, it will equally rebalance each problemname.
    """
    return accuracy_by_group(df,
                             by=by,
                             rebalance=rebalance,
                             accfield='bias')


def accuracy_by_group(df,
                      by=['modelname','prompting','problemname','problemsize','complexity','bintern'],
                      rebalance=True,
                      accfield='acc'):
    """Returns a dataframe containing mean,var,std,sem,count and conf95 of accuracy (the 'acc' field) within each group.
    With rebalance=False this is the direct result of a mere groupby.aggregate
    With rebalance=True the call does equal rebalancing of problemsize (if aggregating across problemsizes)
    Also if aggregating on problemname, it will equally rebalance each problemname.
    """
    df_keys = list(df.keys())

    if not rebalance:
        stats = df.groupby(by)[accfield].aggregate(['mean','sem','count','var','std'])
    else:
        rebalance_fields = ['problemsize','problemname']
        rebalance_fields = [f for f in rebalance_fields if (f not in by and f in df_keys)]
        # print("rebalance_fields:", rebalance_fields)
        newby = by + rebalance_fields
        # print("Initial newby:", newby)
        stats = df.groupby(newby)[accfield].aggregate(['mean','sem','count','var','std'])
        for rebf in rebalance_fields:
            newby = [f for f in newby if f!=rebf]
            # print("    newby:", newby)
            stats = regroup(newby, stats)

    stats['conf95'] = 1.96 * stats['sem']
            
    return stats


def pivot_pretty(
    stats,
    columns=[],
    index=None,
    hide_uniques=True,
    cell_style="str",
    order_prefs=ORDER_PREFS,
    scale=100.,
    valname='value'):
    """Returns a pivot table based on stats 'mean' and 'sem'
    The content of each cell of the resulting table depends on the chosen cell_style
      default cell_style="str" will result in each cell being a *string*
      containing a prettyfied rendering of the mean together with the
      corresponding 95% confidence interval.
    Onther kinds of cell content (including mean, and conficence interval as numerical values)
     can be chosen by specifying a different cell_style (see function format_cell for possibilities)
    If scale=100 (the default) value and sem are first multiplied by 100 to get percentages.
    """

    if isinstance(columns, str):
        columns = [columns]
    st = stats.copy()
    st[valname] = st.apply(lambda x: format_cell(x['mean'],x['sem'],cell_style,scale=scale), axis=1)
    autoindex = [f for f in st.index.names if f not in columns]
    st = st.reset_index()

    reorder_categoricals_inplace(st, order_prefs)

    autoindex = [f for f in autoindex if len(st[f].unique()) > 1]

    if index is None:
        index = autoindex
    # print(st.keys())
    piv = st.pivot_table(values=valname, index=index, columns=columns, aggfunc='first')
    return piv


# Main formatting function for outputting error percentage and confidence typically formatted as a single string for table cells

def format_cell(mean, sem, cell_style="str", scale=100.):
    """
    If scale=100 then mean and sem will first be multiplied by 100 to get percentages.
    cell_style can be one of:
      'mean': returns the mean value as is
      'conf': returns 95% confidence interval (computed as 1.96 * sem)
      'tuple': returns a tuple (mean, conf)
      'str': returns a formated string combinig mean and conf like "mean (conf)"
      'latexstr': returns a formated string combinig mean and conf like "mean {\tiny $\pm conf$}"
    """
    # return f"{(100*mean):.1f}"
    # with confidence
    val = scale * mean
    conf = scale * 1.96 * sem
    if cell_style == "mean":
        return val
    elif cell_style == "conf":
        return conf
    elif cell_style == "tuple":
        return (val, conf)
    elif cell_style == "str":
        if scale==100:
            val_str = f"{val:.1f}"
            conf_str = f"{conf:.1f}"
        elif scale==1:
            val_str = f"{val:.2f}"
            conf_str = f"{conf:.2f}"
        return val_str + " (" + conf_str  + ")"
        # return f"{(100*mean):.1f} ({(100*1.96*sem):.1f})"
    elif cell_style == "latex":
        if scale==100:
            val_str = f"{val:.1f}"
            conf_str = f"{conf:.1f}"
        elif scale==1:
            val_str = f"{val:.2f}"
            conf_str = f"{conf:.2f}"
        return val_str + r"{\tiny $\pm " + conf_str + r"$}"
    else:
        raise ValueError(f"Invalid cell_style: {cell_style}")


############################################################
############################################################
##
##  Generically useful pandas statistical analysis functions
##
############################################################
############################################################

def transform_into_weighted_mean(df, by, fields, weightfield):
    """
    Computes a weighted mean of field(s) within each group obtained via df.groupby(by)
    fields can be a single fieldname or a list of fieldnames
    In each row of a group, the specified fields' values will be replaced by their weighted mean across the rows of the group.
    Note: like other pandas transform, this call returns a df of the same length as the original,
    so the mean will be replicated in each row of the group, replacing the original value of the field
    Returns the new transformed df  (df is not modified by this call)
    """
    if not isinstance(fields, list):
        fields = [fields]
    df = df.copy()
    for field in fields:
        df[field] = df[field] * df[weightfield]
    gb = df.groupby(by)
    weightsum = gb[weightfield].transform('sum').astype(float)
    for field in fields:
        df[field] = gb[field].transform('sum') / weightsum
    return df


# Code for tracking effective counts, and computing properly reweighted means, var, std, sem
def regroup(by, stats, rebalance=True):
    """
    This call takes a dataframe with already computed stats 'mean' and 'var' and associated 'count' as obtained at some finer grained grouping level, and computes the stats for a coarser grouping defined via the 'by' parameter. It outputs usual stats 'mean', 'var', 'std', 'count' and 'sem' (and sum_sq and mean_sq that are used internally for the computation). 
    The main interest of this function, is that it will compute the var and associated std and sem correctly, implicitly based on all the lowest level data points (i.e. var won't be merley the variance of the means in the provided stats parameters, but the correct variance of the lowest level datapoint values).
    Moreover if rebalance is set to True, it will rebalance the counts at the level of the provided stats before computing the new coarser grained statistics. This is achieved simply by setting the effective count to be the minimum count across entries within a group (so that all entries witihn a group get equal count). This will result in computing a now equally reweighted average 'mean' and an appropriately estimated sem. 
    """
    st = stats.reset_index()
    st['mean_sq'] = st['var'] + st['mean'].pow(2)
    if rebalance:
        # st['count'] = st.groupby(by)['count'].transform(lambda x: x * (float(x.min()) / float(x.max())))
        st['count'] = st.groupby(by)['count'].transform(lambda x: float(x.min()))
    st['sum'] = st['mean'] * st['count']
    st['sum_sq'] = st['mean_sq'] * st['count']
    #display(st)
    newst = st.groupby(by)[['count','sum','sum_sq']].sum()
    counts = newst['count'].astype(float)
    newst['mean'] = newst['sum'] / counts
    newst['mean_sq'] = newst['sum_sq'] / counts
    newst['var'] = newst['mean_sq'] - newst['mean'].pow(2)
    newst['std'] = newst['var'].pow(0.5)
    newst['sem'] = newst['std'] / counts.pow(0.5)
    return newst



############################################################
############################################################
##
##  Adding derived fields for more finegrained analysis
##
############################################################
############################################################

def add_bintern_and_globcomplex_fields(df):
    """Appends fields
    - bintern  ('bin' for binary query, 'tern' for ternary query)
    - globcomplex (global complexity as an integer between 0 and 2)
    based on already present query_len and complexity fields
    """
    df['bintern'] = df['query_len'].map({2:'bin', 3:'tern'})
    df['globcomplex']=(df['complexity'].map({'Complexity_0':0,'Complexity_1':1,'Complexity_2':2}) 
                       + df['query_len'] - 2).clip(upper=2)


def add_problem_categorization_fields(df):
    """
    Appends fields:
    - problem_category ('inference', 'consistency', 'completeness')
    - problem_normaltrivial ('normal', 'trivial')
    """
    df["is_control_experiment"] = df["problemname"].isin(["trivial-complete-PI", "incomplete-easy", 
                                                          "trivial-ulist-complete-TF", "trivial-binrel-complete-TF" ])
    map_problem_to_category = {"complete-TF": "inference",
                            "trivial-ulist-complete-TF": "inference",
                            "trivial-binrel-complete-TF": "inference",
                            "complete-PI": "consistency",
                            "trivial-complete-PI": "consistency",
                            "incomplete-easy": "completeness",
                            "incomplete-hard": "completeness",
                            }

    map_problem_to_normaltrivial = {"complete-TF": "normal",
                            "trivial-ulist-complete-TF": "trivial",
                            "trivial-binrel-complete-TF": "trivial",
                            "complete-PI": "normal",
                            "trivial-complete-PI": "trivial",
                            "incomplete-easy": "trivial",
                            "incomplete-hard": "normal",
                           }


    df["problem_category"] = df["problemname"].map(map_problem_to_category)
    df["problem_normaltrivial"] = df["problemname"].map(map_problem_to_normaltrivial)


def add_skin_categorization_fields(df):
    """Appends the folowin fields, that are different caregorizations based on the 'skin':
        - domain : 'salar, spatial, temporal
        - intraining: ind, ood, memorization
    """
    
    # SCALAR
    scalarood = ['rich/poor','stronger/weaker']
    scalarind = ['old/young','tall/short','faster/slower','happier/sadder']
    # Exact repetitions of binary/ternary operators as in the training set
    scalar_memorization = ['old/young/cats', 'tall/short/buildings',
                   'fast/slow/horses', 'happy/sad/dog', 'Heavy/Light/furniture',
                   'High/Low/mountains', 'Big/Small/boxes', 'warm/cool/rivers',
                   'loud/quiet/neighbors']
    scalar = scalarind + scalarood + scalar_memorization

    # SPATIAL 
    spatialind = []  
    spatialood = ['obj_LeftRight','people_LeftRight','obj_FrontBack', 'people_FrontBack','obj_AboveBelow','people_AboveBelow']
    spatial = spatialind + spatialood


    # TEMPORAL
    temporalood = ['courses','shopping']
    temporalind=['trips','activities','conversation','race']
    # Exact repetitions of binary/ternary operators as in the training set
    temporal_memorization = ['trips_countries', 'activities_theater',
                             'conversation_colleagues', 'race_car', 'grocery_store_list',
                             'birthorder', 'plantation', 'renovation','detective']
    # attention, 'detective' appears in some test results, but it should not be analyzed as it is now also part of the fine tuning and memorization set
    # a detective skin filter is added in the relevant sections below
    temporal = temporalind + temporalood + temporal_memorization

    ind = scalarind + spatialind + temporalind
    ood = scalarood + spatialood + temporalood
    memorization = scalar_memorization + temporal_memorization

    # add domain
    map_skin_to_domain = {**{skin: "scalar" for skin in scalar}, 
                          **{skin: "spatial" for skin in spatial},
                          **{skin: "temporal" for skin in temporal},
                         }
    df["domain"] = df['skin'].map(map_skin_to_domain)

    # add intraining (indicate whether the skin was "ind", "ood" or "memorization")
    map_skin_to_intraining = {**{skin: "ind" for skin in ind}, 
                              **{skin: "ood" for skin in ood},
                              **{skin: "memorization" for skin in memorization},
                             }
    df['intraining'] = df['skin'].map(map_skin_to_intraining)   
    
    
def add_fields_for_finegrained_analysis(df):
    """Appends the following fields:
    - bintern  ('bin' for binary query, 'tern' for ternary query)
    - globcomplex (global complexity as an integer between 0 and 2)
    - problem_category ('inference', 'consistency', 'completeness')
    - problem_normaltrivial ('normal', 'trivial')
    - domain ('scalar', 'spatial', 'temporal')
    - intraining ('ind', 'ood', 'memorization')
    """
    add_bintern_and_globcomplex_fields(df)
    add_problem_categorization_fields(df)
    add_skin_categorization_fields(df)
