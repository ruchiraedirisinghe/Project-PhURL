'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Model Build - Random Forest ####

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv('../preprocessed_dataset.csv')
print(dataset.describe())

# remove first 2 columns
X = dataset.iloc[:,2:].values

# assign to 2nd column
y = dataset.iloc[:,1].values 

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Training & Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

## Model Building
# Random Forest Model - Base Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model Evaluation
trainAS = rf.score(X_train, y_train)
val_accurcy = rf.score(X_test, y_test)
val_loss = metrics.log_loss(y_test, y_pred_rf)
precision = metrics.precision_score(y_test, y_pred_rf, average='weighted')
f1 = metrics.f1_score(y_test, y_pred_rf, average='weighted')
r2 = metrics.r2_score(y_test, y_pred_rf)

# Output
print("\nAccuracy on the train dataset: %.4f%%" % (trainAS * 100))
print("Validation Accuracy: %.4f%%" % (trainAS * 100))
print("Validation Loss: %.4f%%" % (trainAS * 100))
print("Precision: %.4f%%" % (trainAS * 100))
print("F1 Score: %.4f%%" % (trainAS * 100))
print("R2 Score: %.4f%%" % (trainAS * 100))







## Figure Generate 13: Confusion Matrix of Random Forest Model

cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame (cm,
                      index = ['benign', 'defacement', 'phishing', 'malware'],
                      columns = ['benign', 'defacement', 'phishing', 'malware'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()




## Figure Generate 14: Feature importances of Random Forest Model

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind="barh",figsize=(10,6))


