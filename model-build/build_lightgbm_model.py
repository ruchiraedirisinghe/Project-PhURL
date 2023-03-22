'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Model Build - LightGBM Classifier ####

'''
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

# Load dataset
dataset = pd.read_csv('../preprocessed_dataset.csv')
print(dataset.describe())

# remove first column
X = dataset.iloc[:,1:].values

# assign to 1st column
y = dataset.iloc[:,1].values 

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Training & Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(X_train, y_train)

y_pred_lgb = LGB_C.predict(X_test)
print(classification_report(y_test,y_pred_lgb,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_lgb)
print("accuracy:   %0.3f" % score)

# calculate additional metrics
trainAS = LGB_C.score(X_train, y_train) 
val_accurcy = LGB_C.score(X_test, y_test)
val_loss = 1 - val_accurcy
precision = metrics.precision_score(y_test, y_pred_lgb, average='macro')
f1 = metrics.f1_score(y_test, y_pred_lgb, average='macro')
r2 = metrics.r2_score(y_test, y_pred_lgb)

# Output
print("\nAccuracy on the train dataset: %.4f%%" % (trainAS * 100))
print("Validation Accuracy: %.4f%%" % (val_accurcy * 100))
print("Validation Loss: %.4f%%" % (val_loss * 100))
print("Precision: %.4f%%" % (precision * 100))
print("F1 Score: %.4f%%" % (f1 * 100))
print("R2 Score: %.4f%%" % (r2 * 100))
