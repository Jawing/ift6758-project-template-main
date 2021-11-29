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
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# Splitting the dataset into the Training set and Test set for grid search CV
#preprocess
#load preprocessed data
df_prep = pickle.load( open("./data/data_test_rg_prep.pickle",'rb'))
y = df_prep['isGoal']
X = df_prep.drop(['isGoal'], axis=1)

#load models for ensembling (assume pretrained)
# search_mlp = load('./models/Q6mlp_s.joblib')
# search_ada = load('./models/Q6ada_s.joblib')
# search_rf = load('./models/Q6rf_s.joblib')
# search_logR = load('./models/Q6logR_s.joblib')

# clf_list = [search_rf,search_mlp,search_logR,search_ada]

# #model pipeline
# search_ens = VotingClassifier(
#     estimators=[
#         ('1', search_rf), # rf
#         ('2', search_mlp), # nn
#         ('3', search_logR), # Logistic Regression
#         ('4', search_ada), # adaboost
#         ],
#     weights=[1, 1, 1, 1],
#     voting='soft')

# search_ens.estimators_ = clf_list
# search_ens.le_ = LabelEncoder().fit(y)
# search_ens.classes_ = search_ens.le_.classes_

#load model
search_ens = load('./models/Q6ens_s.joblib')

y_pred = search_ens.predict(X)
#print("\nResults\nConfusion matrix \n {}".format(confusion_matrix(y_test, y_pred)))
#print(confusion_matrix(y_test, y_pred))
experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y, y_pred))

#plot roc
# from sklearn.metrics import RocCurveDisplay
# svc_disp = RocCurveDisplay.from_estimator(search_mlp, X, y, estimator_name = 'Ensemble')
# plt.savefig('./figures/q7m5_ens_ROC.png')
# plt.show()


from sklearn import metrics
y_score = search_ens.predict_proba(X)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y, y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='Ensemble')
display.plot()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("ROC of Voting Ensemble")
plt.savefig('./figures/q7m5_ens_rg_ROC.png')
plt.show()


#model percentile
lr_probs = search_ens.predict_proba(X)
n = len(lr_probs)
x_axis = np.arange(n)[::-1]*(100/n)
#print(x_axis)

# print(lr_probs)
lr_probs_y = lr_probs[:, 1]
lr_probs_y[::-1].sort()
# print(sum(lr_probs_y))
#print(lr_probs_y)
lr_probs_y_sum = np.cumsum(lr_probs_y)
#print(lr_probs_y_sum)

#goal rate
plt.figure()
plt.plot(
    x_axis,
    lr_probs_y,
)
plt.xlim([100, 0])
plt.ylim([0.0, 1.0])
plt.xlabel("Shot prob model percentile")
plt.ylabel("Goals / (Shots + Goals)")
plt.title("Goal Rate of Ensemble")
plt.savefig('./figures/q7m5_ens_rg_GR.png')
plt.show()

#cumulative plot
plt.figure()
plt.plot(
    x_axis,
    lr_probs_y_sum/sum(lr_probs_y),
)
plt.xlim([100, 0])
plt.ylim([0.0, 1.0])
plt.xlabel("Shot prob model percentile")
plt.ylabel("Proportion")
plt.title("Cumulative % of goals")
plt.savefig('./figures/q7m5_ens_rg_CP.png')
plt.show()

from sklearn.calibration import CalibrationDisplay
disp = CalibrationDisplay.from_estimator(search_ens, X, y, name='Ensemble')
plt.title("Reliability diagram")
plt.savefig('./figures/q7m5_ens_rg_RD.png')
plt.show()


#quantitative metrics
f1 = f1_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))

#these will be logged to your sklearn-demos project on Comet.ml
params={"random_state":0,
        "model_type":"ens",
}

metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X)
experiment.log_parameters(params)
experiment.log_metrics(metrics)
#experiment.end()