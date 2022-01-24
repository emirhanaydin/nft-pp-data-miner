import pandas as pd
from pandas import DataFrame


def cat_to_num_filter(df: DataFrame):
    categorical_cols = df.select_dtypes(['object']).columns
    result = df.copy()
    result[categorical_cols] = result[categorical_cols].apply(lambda x: pd.factorize(x)[0])
    return result
