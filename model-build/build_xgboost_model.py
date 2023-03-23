'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Model Build - XGBoost ####

'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)
print(classification_report(y_test,y_pred_x, target_names=['benign', 'defacement','phishing','malware'], labels=[0,1,2,3]))

score = metrics.accuracy_score(y_test, y_pred_x)
print("accuracy:   %0.3f" % score)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_x)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['benign', 'defacement','phishing','malware'], yticklabels=['benign', 'defacement','phishing','malware'])
plt.title('Confusion matrix for XGBoost Classifier')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Calculate TP, TN, FP, FN
TP = round(cm_norm[1, 1]*100, 4)
TN = round(cm_norm[0, 0]*100, 4)
FP = round(cm_norm[0, 1]*100, 4)
FN = round(cm_norm[1, 0]*100, 4)

print("\nTrue Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

# Feature Importance Plot
plt.figure(figsize=(8,6))
xgb.plot_importance(xgb_c)
plt.title('Feature Importance Plot for XGBoost Classifier')
plt.show()

# ROC Curve Plot
y_prob_x = xgb_c.predict_proba(X_test)
fpr_x, tpr_x, thresholds_x = metrics.roc_curve(y_test, y_prob_x[:,1], pos_label=1)
roc_auc_x = metrics.auc(fpr_x, tpr_x)
plt.figure(figsize=(8,6))
plt.plot(fpr_x, tpr_x, color='orange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_x)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Classifier')
plt.legend(loc="lower right")
plt.show()

# Model Evaluation
trainAS = xgb_c.score(X_train, y_train)
val_accurcy = xgb_c.score(X_test, y_test)
val_loss = 1 - val_accurcy
precision = metrics.precision_score(y_test, y_pred_x, average='weighted')
f1 = metrics.f1_score(y_test, y_pred_x, average='weighted')
r2 = metrics.r2_score(y_test, y_pred_x)

# Output
print("\nAccuracy on the train dataset: %.4f%%" % (trainAS * 100))
print("Validation Accuracy: %.4f%%" % (val_accurcy * 100))
print("Validation Loss: %.4f%%" % (val_loss * 100))
print("Precision: %.4f%%" % (precision * 100))
print("F1 Score: %.4f%%" % (f1 * 100))
print("R2 Score: %.4f%%" % (r2 * 100))