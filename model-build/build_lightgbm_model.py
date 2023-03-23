'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Model Build - LightGBM Classifier ####

'''

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# import joblib

# Load dataset
dataset = pd.read_csv('../preprocessed_dataset.csv')
print(dataset.describe())

# remove first column
X = dataset.iloc[:,1:].values

# assign to 1st column
y = dataset.iloc[:,0].values 

num_features = X.shape[1]

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Training & Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

num_classes = len(le.classes_)

lgb = LGBMClassifier(objective='multiclass', boosting_type='gbdt', n_jobs=5, 
                     silent=True, random_state=5, num_class=num_classes, num_leaves=num_features)
LGB_C = lgb.fit(X_train, y_train)

y_pred_lgb = LGB_C.predict(X_test)
print(classification_report(y_test, y_pred_lgb, target_names=['benign', 'defacement','phishing','malware']))

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



# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred_lgb)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# # Plotting ROC Curve
# y_pred_proba_lgb = LGB_C.predict_proba(X_test)
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba_lgb[:,1], pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# import lightgbm as lgb
# import matplotlib.pyplot as plt

# # Train the LGBMClassifier model
# LGB_C = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31)
# LGB_C.fit(X_train, y_train)

# # Plot feature importance
# lgb.plot_importance(LGB_C, max_num_features=21, height=0.8)
# plt.show()


# Calculate TP, TN, FP, FN
conf_matrix_normlized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
TP = round(conf_matrix_normlized[1, 1]*100, 4)
TN = round(conf_matrix_normlized[0, 0]*100, 4)
FP = round(conf_matrix_normlized[0, 1]*100, 4)
FN = round(conf_matrix_normlized[1, 0]*100, 4)

print("\nTrue Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)


# # Save the model as a joblib file
# joblib.dump(LGB_C, 'lgb_model.joblib')