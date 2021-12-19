
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
from joblib import dump, load

#import plot tools
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

#local imports
from ift6758.data.functions import loadstats
from ift6758.data.tidyData_adv import tidyData_adv

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve, CalibrationDisplay
from xgboost import XGBClassifier  
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime



dfs_2015 = loadstats(2015,'./data/', playoffs=False,regular=True)
df_2015 = tidyData_adv(dfs_2015)

dfs_2016 = loadstats(2016,'./data/', playoffs=False,regular=True)
df_2016 = tidyData_adv(dfs_2016)

dfs_2017 = loadstats(2017,'./data/', playoffs=False,regular=True)
df_2017 = tidyData_adv(dfs_2017)

dfs_2018 = loadstats(2018,'./data/', playoffs=False,regular=True)
df_2018 = tidyData_adv(dfs_2018)

# dfs_2019 = loadstats(2019,'./data/')
# df_2019 = tidyData_adv(dfs_2019) <---- do this on section# 7

df = df_2015.append(df_2016, ignore_index=True).append(df_2017, ignore_index=True).append(df_2018, ignore_index=True)

data_angle_distance = df[["dist_goal","angle_goal", "isGoal"]].dropna()

X_angle_distance_xg = data_angle_distance[["dist_goal","angle_goal"]].astype('float')
y_angle_distance_xg = data_angle_distance["isGoal"].apply( lambda x : 1 if x else 0 )

X_train_ad_xg, X_test_ad_xg, y_train_ad_xg, y_test_ad_xg = train_test_split(X_angle_distance_xg, 
                                                                            y_angle_distance_xg, test_size=0.33, 
                                                                            random_state=42)



# search_xgboost_tune = pickle.load(open('./models/Q51_XGboost_distance_angle.joblib', 'rb'))

search_xgboost_tune = load('./models/Q51_XGboost_distance_angle.joblib')

y_pred = search_xgboost_tune.predict(X_test_ad_xg)

experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y_test_ad_xg, y_pred))


#quantitative metrics
f1 = f1_score(y_test_ad_xg, y_pred)
precision = precision_score(y_test_ad_xg, y_pred)
recall = recall_score(y_test_ad_xg, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))


metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train_ad_xg)
experiment.log_metrics(metrics)
experiment.log_model("Q5_XGBOOST", './models/Q51_XGboost_distance_angle.joblib')