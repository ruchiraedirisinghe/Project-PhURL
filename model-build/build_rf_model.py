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

# remove first column
X = dataset.iloc[:,1:].values

# assign to 1st column
y = dataset.iloc[:,0].values 

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
val_loss = 1 - val_accurcy
precision = metrics.precision_score(y_test, y_pred_rf, average='weighted')
f1 = metrics.f1_score(y_test, y_pred_rf, average='weighted')
r2 = metrics.r2_score(y_test, y_pred_rf)

# Output
print("\nAccuracy on the train dataset: %.4f%%" % (trainAS * 100))
print("Validation Accuracy: %.4f%%" % (val_accurcy * 100))
print("Validation Loss: %.4f%%" % (val_loss * 100))
print("Precision: %.4f%%" % (precision * 100))
print("F1 Score: %.4f%%" % (f1 * 100))
print("R2 Score: %.4f%%" % (r2 * 100))



# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
conf_matrix_normlized = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normlized, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
y_pred_proba = rf.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
plt.plot(fpr[0], tpr[0], label='Class 0 ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], label='Class 1 ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='Class 2 ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Generate Feature Importance Plot
feat_importances = pd.Series(rf.feature_importances_, index=dataset.columns[1:])
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()


# Calculate TP, TN, FP, FN
TP = round(conf_matrix_normlized[1, 1]*100, 4)
TN = round(conf_matrix_normlized[0, 0]*100, 4)
FP = round(conf_matrix_normlized[0, 1]*100, 4)
FN = round(conf_matrix_normlized[1, 0]*100, 4)

print("\nTrue Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
