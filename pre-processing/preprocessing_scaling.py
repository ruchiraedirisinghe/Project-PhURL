'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Preprocessing - Standard Scaling ####

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib


# Loading Datasets

df=pd.read_csv('../malicious_phish.csv')

print(df.shape)
df.head()

df.type.value_counts()

# remove first 2 columns
X = dataset.iloc[:,2:].values

# assign to 2nd column
y = dataset.iloc[:,1].values 


# Initialize the scaler
scale = StandardScaler()

# Fit the scaler to the data
scale.fit(X)

# Save the scaler as a joblib file
joblib.dump(scale, 'scaledfeature.joblib')


