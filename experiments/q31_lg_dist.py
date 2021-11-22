import os

import numpy as np 
import pandas as pd
import seaborn as sns

from comet_ml import Experiment
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from ift6758.data.functions import loadstats
from ift6758.data.tidyData_adv import tidyData_adv


dfs_2015 = loadstats(2015,'./data/')
df_2015 = tidyData(dfs_2015)

dfs_2016 = loadstats(2016,'./data/')
df_2016 = tidyData(dfs_2016)

dfs_2017 = loadstats(2017,'./data/')
df_2017 = tidyData(dfs_2017)

dfs_2018 = loadstats(2018,'./data/')
df_2018 = tidyData(dfs_2018)

#dfs_2019 = loadstats(2019,'./data/')
#df_2019 = tidyData(dfs_2019) <---- do this on section# 7

df = df_2015.append(df_2016, ignore_index=True).append(df_2017, ignore_index=True).append(df_2018, ignore_index=True).dropna()

data_angle = df[["angle_goal", "isGoal"]].dropna()

X = data_distance[["dist_goal"]]
y = data_distance["isGoal"].apply( lambda x : 1 if x else 0 )


experiment = Experiment( project_name="milestone-2" )

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train_d, y_train_d)

y_pred = clf.predict(X_test_d)

clf.predict_proba(X_test_d)

f1 = f1_score(y_test_d, y_pred)
precision = precision_score(y_test_d, y_pred)
recall = recall_score(y_test_d, y_pred)

metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train)
experiment.log_metrics(metrics)



