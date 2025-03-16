import numpy as np
import pandas as pd

data = {
    'Indicator A': [],
    'Indicator B': [],
    'Indicator C': [],
    'Indicator D': [],
}

df = pd.DataFrame(data)
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())
normalized_df = normalize(df)
std_dev = normalized_df.std()
corr_matrix = normalized_df.corr()
info_content = std_dev * ((1 - corr_matrix).sum(axis=1))
weights = info_content / info_content.sum()

print(weights)