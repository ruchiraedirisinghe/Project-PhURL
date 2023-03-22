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


# Training & Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)




