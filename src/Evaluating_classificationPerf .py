# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:40:13 2020

@author: adds0
"""
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
labels = pd.read_csv('D:/OneDrive - City, University of London/MSc Data Science City/Individual Project/GANKneeMRI2/src/model_weights/class/wgan with [1,0,1]/labels.csv')
print(labels.head())
preds = pd.read_csv('D:/OneDrive - City, University of London/MSc Data Science City/Individual Project/GANKneeMRI2/src/model_weights/class/wgan with [1,0,1]/preds.csv')


# =============================================================================
#                         # Confusion matrix
# =============================================================================
abnormal_label = labels['abnormal']
abnormal_pred = preds['abnormal']
abnormal_label_n = np.array(abnormal_label)
abnormal_pred_n = np.array(abnormal_pred)
abnormal_pred_n = np.where(abnormal_pred_n <= 0.5, 0, abnormal_pred_n)
abnormal_pred_n = np.where(abnormal_pred_n > 0.5, 1, abnormal_pred_n)
#abnormal_pred_n = abnormal_pred_n. astype(int)
#Abnormal Confusion matrix
ab_cm = confusion_matrix(abnormal_label_n, abnormal_pred_n)
import seaborn as sns
plt.figure(figsize=(10,10))
ax= plt.subplot()
sns_plot = sns.heatmap(ab_cm, cmap="Blues", annot=True, ax = ax,annot_kws={"size": 24},fmt=".1f"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
# ACL confusion matrix
acl_label = labels['acl']
acl_pred = preds['acl']
acl_label_n = np.array(acl_label)
acl_pred_n = np.array(acl_pred)
acl_pred_n = np.where(acl_pred_n <= 0.5, 0, acl_pred_n)
acl_pred_n = np.where(acl_pred_n > 0.5, 1, acl_pred_n)
acl_cm = confusion_matrix(acl_label_n, acl_pred_n)

plt.figure(figsize=(10,10))
ax= plt.subplot()
sns_plot = sns.heatmap(acl_cm,cmap="Blues", annot=True, ax = ax,annot_kws={"size": 19},fmt=".1f"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
# Meniscus Confusion matrix
meniscus_label = labels['meniscus']
meniscus_pred = preds['meniscus']
meniscus_label_n = np.array(meniscus_label)
meniscus_pred_n = np.array(meniscus_pred)
meniscus_pred_n = np.where(meniscus_pred_n <= 0.5, 0, meniscus_pred_n)
meniscus_pred_n = np.where(meniscus_pred_n > 0.5, 1, meniscus_pred_n)
meniscus_cm = confusion_matrix(meniscus_label_n, meniscus_pred_n)

plt.figure(figsize=(10,10))
ax= plt.subplot()
sns_plot = sns.heatmap(meniscus_cm,cmap="Blues", annot=True, ax = ax,annot_kws={"size": 19},fmt=".1f"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
# =============================================================================
# #                            ROC CURVE
# =============================================================================
# Abnormal
abnormal_label = labels['abnormal']
abnormal_pred = preds['abnormal']
abnormal_label_n = np.array(abnormal_label)
abnormal_pred_n = np.array(abnormal_pred)
fpr_b, tpr_b, thresholds = metrics.roc_curve(abnormal_label_n, abnormal_pred_n)#pos_label=0
auc = metrics.roc_auc_score(abnormal_label_n, abnormal_pred_n)
# Print ROC curve
plt.plot(fpr_b,tpr_b, '-b', label = 'abnormal AUC: {:.3f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

#ACL 
acl_label = labels['acl']
acl_pred = preds['acl']
acl_label_n = np.array(acl_label)
acl_pred_n = np.array(acl_pred)
fpr_a, tpr_a, thresholds = metrics.roc_curve(acl_label_n, acl_pred_n)#pos_label=0
auc = metrics.roc_auc_score(acl_label_n, acl_pred_n)
# Print ROC curve
plt.plot(fpr_a,tpr_a, '-r', label = 'acl AUC: {:.3f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#meniscus
meniscus_label = labels['meniscus']
meniscus_pred = preds['meniscus']
meniscus_label_n = np.array(meniscus_label)
meniscus_pred_n = np.array(meniscus_pred)
fpr_m, tpr_m, thresholds = metrics.roc_curve(meniscus_label_n, meniscus_pred_n)#pos_label=0
auc = metrics.roc_auc_score(meniscus_label_n, meniscus_pred_n)
# Print ROC curve
plt.plot(fpr_m,tpr_m, '-g', label = 'meniscus AUC: {:.3f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
# =============================================================================
# ## Specificity, Sensitivity, Recall Precision, Fscore, Accuracy
# =============================================================================
# Anormal
abnormal_label = labels['abnormal']
abnormal_pred = preds['abnormal']
abnormal_label_n = np.array(abnormal_label)
abnormal_pred_n = np.array(abnormal_pred)
abnormal_pred_n = np.where(abnormal_pred_n <= 0.5, 0, abnormal_pred_n)
abnormal_pred_n = np.where(abnormal_pred_n > 0.5, 1, abnormal_pred_n)
a = sklearn.metrics.classification_report(abnormal_label_n, abnormal_pred_n)
tp_b, fp_b, fn_b, tn_b = confusion_matrix(abnormal_label_n, abnormal_pred_n).ravel()
b_sensitivity,b_specificity = tp_b/(tp_b+fn_b),tn_b/(tn_b+fp_b)
#ACL
acl_label = labels['acl']
acl_pred = preds['acl']
acl_label_n = np.array(acl_label)
acl_pred_n = np.array(acl_pred)
acl_pred_n = np.where(acl_pred_n <= 0.5, 0, acl_pred_n)
acl_pred_n = np.where(acl_pred_n > 0.5, 1, acl_pred_n)
a_cl = sklearn.metrics.classification_report(acl_label_n, acl_pred_n)
tp_b, fp_b, fn_b, tn_b = confusion_matrix(acl_label_n, acl_pred_n).ravel()
a_sensitivity,a_specificity = tp_a/(tp_a+fn_a),tn_a/(tn_a+fp_a)
# Meniscus
meniscus_label = labels['meniscus']
meniscus_pred = preds['meniscus']
meniscus_label_n = np.array(meniscus_label)
meniscus_pred_n = np.array(meniscus_pred)
meniscus_pred_n = np.where(meniscus_pred_n <= 0.5, 0, meniscus_pred_n)
meniscus_pred_n = np.where(meniscus_pred_n > 0.5, 1, meniscus_pred_n)
m = sklearn.metrics.classification_report(meniscus_label_n, meniscus_pred_n)
tp_b, fp_b, fn_b, tn_b = confusion_matrix(meniscus_label_n, meniscus_pred_n).ravel()
m_sensitivity,m_specificity = tp_m/(tp_m+fn_m),tn_m/(tn_m+fp_m)


# using confidence interval
from math import sqrt
def wilson_score(p, n, z = 1.96):
    denominator = 1 + z**2/n
    centre_probability = p + z*z / (2*n)
    standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_probability - z*standard_deviation) / denominator
    upper_bound = (centre_probability + z*standard_deviation) / denominator
    return (lower_bound, upper_bound)

positive = 102
total = 120
p  = positive / total
(p, wilson_score(0.69, total))