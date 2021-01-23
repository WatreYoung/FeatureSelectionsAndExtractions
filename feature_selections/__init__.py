# -*- conding:utf-8 -*-
# file       : __init__.py.py
# time       : 2020/5/23 11:27 AM
# author     : littlely
# description: 
import pandas as pd

from feature_selections import mrmr

if __name__ == '__main__':
    df = pd.read_csv("./iris.csv", header=None, names=['a', 'b', 'c', 'd', 'label'])
    # print(df.head())
    features = df[['a', 'b', 'c', 'd']]
    # print(features.head())
    # print(features.values)
    labels = df["label"]
    labels = labels.map(lambda x: 0 if x == "setosa" else 1 if x == "versicolor" else 2)
    # print(labels.values)
    # print(labels)

    m = mrmr.MRMR(1)
    m.fit(features.values[1:], labels.values[1:])
    x = m.transform(features.values)
    # print(x)
    print(m.important_features)