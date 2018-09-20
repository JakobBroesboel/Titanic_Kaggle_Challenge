import numpy as np
import csv
import pandas as pd

# 0           1        2      3    4   5   6     7     8      9    10
#[PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin]
myfilter = [1,2,4,5,6,7,9]

def load_data(file, filter):
    return pd.read_csv(file, usecols = filter)

df = load_data("data/train.csv", myfilter)
df = df[pd.notnull(df['Age'])]
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

df.to_csv("data/processed_train.csv", index = False)
