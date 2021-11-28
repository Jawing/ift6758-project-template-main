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
# import plotly.express as px

#local imports
from ift6758.data.functions import loadstats
from ift6758.data.tidyData_adv import tidyData_adv

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve, CalibrationDisplay

dfs_2015 = loadstats(2015,'./data/')
df_2015 = tidyData_adv(dfs_2015)

dfs_2016 = loadstats(2016,'./data/')
df_2016 = tidyData_adv(dfs_2016)

dfs_2017 = loadstats(2017,'./data/')
df_2017 = tidyData_adv(dfs_2017)

dfs_2018 = loadstats(2018,'./data/')
df_2018 = tidyData_adv(dfs_2018)

# dfs_2019 = loadstats(2019,'./data/')
# df_2019 = tidyData_adv(dfs_2019) <---- do this on section# 7

df = df_2015.append(df_2016, ignore_index=True).append(df_2017, ignore_index=True).append(df_2018, ignore_index=True)

data_distance = df[["dist_goal", "isGoal"]].dropna()

X = data_distance[["dist_goal"]]
y = data_distance["isGoal"].apply( lambda x : 1 if x else 0 )
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0).fit(X_train_d, y_train_d)
# y_pred = clf.predict(X_test_d)
# print(clf.predict_proba(X_test_d))
# clf.score(X_test_d, y_test_d)

# print("The classification report is as follows:\n", classification_report(y_test_d,y_pred))

# f1 = f1_score(y_test_d, y_pred)
# precision = precision_score(y_test_d, y_pred)
# recall = recall_score(y_test_d, y_pred)

# # Saving the model 

# # save the classifier
# location = os.getcwd() +'/models'
# fullpath = os.path.join(location)
# with open(fullpath + '/Q3.1_log_reg_distance.pkl', 'wb') as fid:
#     pickle.dump(clf, fid) 

#load model pickle.load(open('model.pkl', 'rb'))
search_logreg = pickle.load(open('./models/Q31_log_reg_distance.pkl', 'rb'))

y_pred = search_logreg.predict(X_test_d)

experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y_test_d, y_pred))


#quantitative metrics
f1 = f1_score(y_test_d, y_pred)
precision = precision_score(y_test_d, y_pred)
recall = recall_score(y_test_d, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))


metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train_d)
experiment.log_metrics(metrics)
experiment.log_model("Q3_LogReg_Distance", './models/Q31_log_reg_distance.pkl')