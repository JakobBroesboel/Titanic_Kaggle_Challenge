import numpy as np
import pandas as pd

def get_data(file):
    df = pd.read_csv("data/processed_train.csv")
    return df.values[:,1:], df.values[:,0]

data, labels = get_data("data/processed_train.csv")

print data.tolist()
