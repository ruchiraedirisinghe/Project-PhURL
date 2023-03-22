'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Preprocessing - Dataset Preprocessing ####

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
dataset = pd.read_csv('processedurl.csv')
print(dataset.describe())

# remove first 2 columns
X = dataset.iloc[:,2:].values

# assign to 2nd column
y = dataset.iloc[:,1].values 


## Label Encording

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Feature Scaling - Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Create a new DataFrame with preprocessed data
preprocessed_df = pd.DataFrame(data=X, columns=dataset.columns[1:-1])

# Add the classification column to the preprocessed DataFrame
preprocessed_df['type'] = y

# Save preprocessed dataset as CSV file
preprocessed_df.to_csv('preprocessed_dataset.csv', index=False)
print("Dataset Creation Successfully Completed!!!")



