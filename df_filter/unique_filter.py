from pandas import DataFrame

excluded = [
    '01 Buster',
    '838 Spyder',
    'Aqua Blaster',
    'B.O.X.',
    'B.R.1.C.K',
    'CHMP',
    'Droid Ravager',
    'Drumstick',
    'Grumpii',
    'HBB Renegade',
    'MegaBoidz',
    'Meta',
    'Order 66',
    'Puff Boxer',
    'R.E.X. 02',
    'Red Steel',
    'SB Skyhammer',
    'T.I.G.E.R.Zero',
    'WAT 51',
    '| | | | | | | | | | | | | | | |',
]


def unique_filter(df: DataFrame):
    return df[~df['Front'].isin(excluded)]
