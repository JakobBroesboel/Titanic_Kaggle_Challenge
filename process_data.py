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

#Normalizing values
df['Age'] = df['Age'] / df['Age'].max()
df['SibSp'] = df['SibSp'] / df['SibSp'].max()
df['Parch'] = df['Parch'] / df['Parch'].max()
df['Fare'] = df['Fare'] / df['Fare'].max()

cat = pd.get_dummies(df['Pclass'])
df.insert(loc = 1, column = 'FirstClass', value = cat[1])
df.insert(loc = 2, column = 'SecondClass', value = cat[2])
df.insert(loc = 3, column = 'ThirdClass', value = cat[3])

df = df.drop('Pclass', axis=1)


df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

df.to_csv("data/processed_train.csv", index = False)
