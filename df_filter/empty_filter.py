from pandas import DataFrame


def empty_filter(df: DataFrame):
    return df.dropna(subset=['Front'])
