#import model tracking
import comet_ml
comet_ml.init()

from comet_ml import Experiment
experiment = Experiment(
    project_name="milestone-2",
    workspace="binulal",
)


#general imports
import os
import numpy as np 
import pandas as pd

#import plot tools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#local imports
from ift6758.data.functions import loadstats
from ift6758.data.functions import pre_process
from ift6758.data.tidyData_adv import tidyData_adv

import pickle

#import sklearn modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

#save with joblib faster with np
from joblib import dump, load

#import model
from sklearn.svm import SVC

# Splitting the dataset into the Training set and Test set for grid search CV 
#preprocess
#load preprocessed data
df_prep = pickle.load( open("./data/data_train_prep.pickle",'rb'))
y = df_prep['isGoal']
X = df_prep.drop(['isGoal'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)


#model pipeline
search_svc = Pipeline(steps=[('scaler', StandardScaler()), ("pca", PCA(n_components = 85)), ("svc", SVC(kernel='rbf',C=10))])


#save model
#search_svc.fit(X_train, y_train)
#dump(search_svc, './models/Q6svc_s.joblib')

#load model
search_svc = load('./models/Q6svc_s.joblib')

#get confusion matrix and predictions
# print("Best parameter (CV score=%0.3f):" % search_svc.best_score_)
# print(search_svc.best_params_)

y_pred = search_svc.predict(X_test)
#print("\nResults\nConfusion matrix \n {}".format(confusion_matrix(y_test, y_pred)))
#print(confusion_matrix(y_test, y_pred))
experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y_test, y_pred))

#plot roc
from sklearn.metrics import roc_curve, auc
y_score = search_svc.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC of SVC")
plt.legend(loc="lower right")

plt.savefig('./figures/q62_svc_ROC.png')
plt.show()

# #model percentile
# lr_probs = search_svc.predict_proba(X_test)
# n = len(lr_probs)
# x_axis = np.arange(n)[::-1]*(100/n)
# #print(x_axis)

# # print(lr_probs)
# lr_probs_y = lr_probs[:, 1]
# lr_probs_y[::-1].sort()
# # print(sum(lr_probs_y))
# #print(lr_probs_y)
# lr_probs_y_sum = np.cumsum(lr_probs_y)
# #print(lr_probs_y_sum)

# #goal rate
# plt.figure()
# plt.plot(
#     x_axis,
#     lr_probs_y,
# )
# plt.xlim([100, 0])
# plt.ylim([0.0, 1.0])
# plt.xlabel("Shot prob model percentile")
# plt.ylabel("Goals / (Shots + Goals)")
# plt.title("Goal Rate of SVC")
# plt.savefig('./figures/q62_svc_GR.png')
# plt.show()

# #cumulative plot
# plt.figure()
# plt.plot(
#     x_axis,
#     lr_probs_y_sum/sum(lr_probs_y),
# )
# plt.xlim([100, 0])
# plt.ylim([0.0, 1.0])
# plt.xlabel("Shot prob model percentile")
# plt.ylabel("Proportion")
# plt.title("Cumulative % of goals")
# plt.savefig('./figures/q62_svc_CP.png')
# plt.show()

# from sklearn.calibration import CalibrationDisplay
# disp = CalibrationDisplay.from_estimator(search_svc, X_test, y_test, name='SVC')
# plt.title("Reliability diagram")
# plt.savefig('./figures/q62_svc_RD.png')
# plt.show()



#quantitative metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))

#these will be logged to your sklearn-demos project on Comet.ml
params={"random_state":0,
        "model_type":"svc",
        "scaler":"standard scaler",
        "stratify":True
}

metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train)
experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.log_model("Q6_Full_svc", "./models/Q6svc_s.joblib")
#experiment.end()