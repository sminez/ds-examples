# import math
# import numpy as np
import pandas as pd
# import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_CMAP = 'RdYlBu_r'


def show_missing_data(df, cmap=DEFAULT_CMAP):
    '''
    Display a quick heatmap of where we have missing data in a dataframe.
    (If not used in a notebook, this will open a plot window.)
    '''
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    plt.figure(figsize=(16, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cmap=cmap)


def show_percentage_null(df):
    '''
    Find the percentage of nulls in each dataframe column and print them.
    '''
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    for col in df.columns:
        nulls = df[df[col].isnull()]
        if len(nulls) > 0:
            perc = round(len(nulls) / len(df) * 100, 2)
            print("{col}: {perc}% NULL".format(col=col, perc=perc))


def fill_na_with_grouped_means(df, fill_column, grouping_column, dp=2):
    '''
    Compute the grouped means of a `fill_column` when partitioned by
    `grouping_column` and use these to fill any missing values.
    Mean values will be rounded to `dp` decimal places.
    '''
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    means = df.groupby(grouping_column).mean()[fill_column]

    def filler(row):
        target = row.loc[fill_column]
        key = row.loc[grouping_column]
        return means[key] if pd.isnull(target) else target

    df[fill_column] = df.apply(filler, axis=1)
