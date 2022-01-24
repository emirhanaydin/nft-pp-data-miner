from pandas import DataFrame

excluded = [
    "Pilot",
]


def correlation_filter(df: DataFrame):
    return df[df.columns.difference(excluded)]
