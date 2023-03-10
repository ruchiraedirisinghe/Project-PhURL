#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####

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

# drop missing value records
dataset.dropna(inplace=True)

#replace missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(x[:,1:110])
x[:,1:110] = imputer.transform(x[:,1:110])

# Data Encoding: Handle/encode categorical data

# One hot encording
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encorder',OneHotEncoder(),[0])],remainder="passthrough")
x= np.array(ct.fit_transform(x))
print(x)

# Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print(y)