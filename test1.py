#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####

"""
@author: ruchi_lb7de3w
"""

# Importing Data
import numpy as np
import pandas as pd

# loading dataset
dataset = pd.read_csv ('phishing-dataset-variation.csv')
print(dataset.describe())

# Create dependent & independent variable vectors

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)

# handle missing data

# count the number of missing values in each column
print(dataset.isnull().sum())