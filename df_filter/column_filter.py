from pandas import DataFrame


def column_filter(df: DataFrame):
    return df.drop(columns=['Id', 'Serial Number'])
