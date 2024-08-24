import numpy as np
import pandas as pd

def range_sum(l):
    ranges_sum = np.sum([abs(a - b) for a,b in l])
    return ranges_sum

def calculate_coverage(dataset_df, l):
    df_copy = dataset_df.copy()
    for i,feature in enumerate(l):
        df_copy = df_copy[(feature[0] <= df_copy[df_copy.columns[i]]) &  (df_copy[df_copy.columns[i]] <= feature[1])]
    #display(df_copy)
    return df_copy