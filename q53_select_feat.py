
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import calibration_curve, CalibrationDisplay
from xgboost import XGBClassifier  
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

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

data_xgboost = df

#changing categorical 
data_xgboost["rebound"] = data_xgboost["rebound"].apply( lambda x : 1 if x else 0 )
#print(data_xgboost.isna().sum())
data_xgboost = data_xgboost[["periodSeconds", "period", "coordinates_x", "coordinates_y","dist_goal", "angle_goal", 
                             "shotType", "eventType_last", "coordinates_x_last","coordinates_y_last", "distance_last",
                             "periodSeconds_last","rebound","angle_change","speed", "isGoal"]]


data_xgboost_new = data_xgboost.dropna()
print(data_xgboost_new.isna().shape)

X_xg = data_xgboost_new.iloc[:, :-1]
y_xg = data_xgboost_new.iloc[:, -1].apply( lambda x : 1 if x else 0 )
X_train, X_test, y_train, y_test = train_test_split(X_xg, y_xg, test_size=0.33, random_state=42)

df = pd.get_dummies(X_xg[["shotType", "eventType_last"]])
#df 

#Concat new and the previous dataframe
X = pd.concat([X_xg,df], axis = 1)

#dropping the two columns 
X_new = X.drop(['shotType', 'eventType_last'], axis = 1)
print(X_new.shape)

# let's take first 10 most important features to train XG_boost 
X_new_ft = X_new[['dist_goal','coordinates_y', 'periodSeconds_last', 'periodSeconds', 'angle_goal',
              'period', 'speed', 'shotType_Slap Shot', 'shotType_Backhand', 'distance_last', 'shotType_Wrist Shot' ]]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_new_ft.astype('float'), y_xg, test_size=0.33, random_state=42)



search_xgboost_tune = pickle.load(open('./models/Q53_XGboost_feature_tuned.pkl', 'rb'))

y_pred = search_xgboost_tune.predict(X_test_1)

experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y_test_1, y_pred))


#quantitative metrics
f1 = f1_score(y_test_1, y_pred)
precision = precision_score(y_test_1, y_pred)
recall = recall_score(y_test_1, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))


metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train_1)
experiment.log_metrics(metrics)
experiment.log_model("Q5_XGBOOST", './models/Q53_XGboost_feature_tuned.pkl')