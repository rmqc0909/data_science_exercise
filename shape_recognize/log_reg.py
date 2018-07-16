#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 15:30
# @Author  : xiedan
# @File    : log_reg.py
# 逻辑回归模型
from sklearn.linear_model import LogisticRegression
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

train_labels = train.pop('y')

clf = LogisticRegression()
clf.fit(train, train_labels)

submit = pd.read_csv('data/sample_submit.csv')
submit['y'] = clf.predict(test)
submit.to_csv('data/LR_prediction.csv', index=False)
