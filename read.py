import pandas as pd
from sklearn.preprocessing import scale


def read_wine():
    df = pd.read_csv("wineParcial.data", sep=",", header=None)
    df[0] -= 1
    df_class = df[0]
    df = pd.DataFrame(scale(df))

    df.drop(columns=[0], inplace=True)

    return df, df_class
