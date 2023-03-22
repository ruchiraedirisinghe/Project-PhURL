'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Model Build - XGBoost ####

'''


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb


# # Load dataset
# dataset = pd.read_csv('../preprocessed_dataset.csv')
# print(dataset.describe())

# # remove first 2 columns
# X = dataset.iloc[:,2:].values

# # assign to 2nd column
# y = dataset.iloc[:,1].values 

# # Encoding categorical variables
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Training & Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)


# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix


# xgb_c = xgb.XGBClassifier(n_estimators= 100)
# xgb_c.fit(X_train,y_train)
# y_pred_x = xgb_c.predict(X_test)
# print(classification_report(y_test,y_pred_x,target_names=['benign', 'defacement','phishing','malware']))

# score = metrics.accuracy_score(y_test, y_pred_x)
# print("accuracy:   %0.3f" % score)




# ## Figure Generate 15: Confusion Matrix of XGboost Classifier

# cm = confusion_matrix(y_test, y_pred_x)
# cm_df = pd.DataFrame (cm,
#                       index = ['benign', 'defacement', 'phishing', 'malware'],
#                       columns = ['benign', 'defacement', 'phishing', 'malware'])
# plt.figure(figsize=(8,6))
# sns.heatmap(cm_df, annot=True, fmt=".1f")
# plt.title('Confusion Matrix')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()




# ## Figure Generate 16: Feature importances of XGboost Classifier

# feat_importances = pd.Series(xgb_c.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind="barh",figsize=(10,6))








# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb

# # Load dataset
# dataset = pd.read_csv('../preprocessed_dataset.csv')
# print(dataset.describe())

# # remove first 2 columns
# X = dataset.iloc[:,2:].values

# # assign to 2nd column
# y = dataset.iloc[:,1].values 

# # Encoding categorical variables
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Training & Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix

# xgb_c = xgb.XGBClassifier(n_estimators= 100)
# xgb_c.fit(X_train,y_train)
# y_pred_x = xgb_c.predict(X_test)
# print(classification_report(y_test,y_pred_x, target_names=['benign', 'defacement','phishing','malware'], labels=[0,1,2,3]))

# score = metrics.accuracy_score(y_test, y_pred_x)
# print("accuracy:   %0.3f" % score)

# ## Figure Generate 15: Confusion Matrix of XGboost Classifier

# cm = confusion_matrix(y_test, y_pred_x)
# cm_df = pd.DataFrame (cm,
#                       index = ['benign', 'defacement', 'phishing', 'malware'],
#                       columns = ['benign', 'defacement', 'phishing', 'malware'])
# plt.figure(figsize=(8,6))
# sns.heatmap(cm_df, annot=True, fmt=".1f")
# plt.title('Confusion Matrix')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()

# ## Figure Generate 16: Feature importances of XGboost Classifier

# feat_importances = pd.Series(xgb_c.feature_importances_, index=dataset.iloc[:,2:].columns)
# feat_importances.sort_values().plot(kind="barh", figsize=(10,6))

# # Model Evaluation
# trainAS = xgb_c.score(X_train, y_train)
# val_accurcy = xgb_c.score(X_test, y_test)
# val_loss = metrics.log_loss(y_test, y_pred_x)
# precision = metrics.precision_score(y_test, y_pred_x, average='weighted')
# f1 = metrics.f1_score(y_test, y_pred_x, average='weighted')
# r2 = metrics.r2_score(y_test, y_pred_x)

# # Output
# print("\nAccuracy on the train dataset: %.4f%%" % (trainAS * 100))
# print("Validation Accuracy: %.4f%%" % (val_accurcy * 100))
# print("Validation Loss: %.4f%%" % (val_loss * 100))
# print("Precision: %.4f%%" % (precision * 100))
# print("F1 Score: %.4f%%" % (f1 * 100))
# print("R2 Score: %.4f%%" % (r2 * 100))








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)
print(classification_report(y_test,y_pred_x, target_names=['benign', 'defacement','phishing','malware'], labels=[0,1,2,3]))

score = metrics.accuracy_score(y_test, y_pred_x)
print("accuracy:   %0.3f" % score)

## Figure Generate 15: Confusion Matrix of XGboost Classifier

cm = confusion_matrix(y_test, y_pred_x)
cm_df = pd.DataFrame (cm,
                      index = ['benign', 'defacement', 'phishing', 'malware'],
                      columns = ['benign', 'defacement', 'phishing', 'malware'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

## Figure Generate 16: Feature importances of XGboost Classifier

feat_importances = pd.Series(xgb_c.feature_importances_, index=dataset.iloc[:,2:].columns)
feat_importances.sort_values().plot(kind="barh", figsize=(10,6))
