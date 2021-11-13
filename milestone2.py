import os

import numpy as np 
import pandas as pd
import seaborn as sns

from comet_ml import Experiment
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from ift6758.data.functions import loadstats
from ift6758.data.tidyData import tidyData


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

opponent_coordinate = [-90,0]

playerXYCoordDF = df[["coordinates_x","coordinates_y"]]
playerXYCoordDF.head()

playerXY_DF = playerXYCoordDF.apply( lambda x : pd.Series( opponent_coordinate, index = ['opponent_x', 'opponent_y'] ), axis=1, result_type="expand")
playerXY_DF.head()

playerXY_DF2 = playerXYCoordDF.join(playerXY_DF)

# playerXY_DF3 
df["Distance"]  = playerXY_DF2.apply( lambda row : np.sqrt((row.opponent_x - row.coordinates_x)**2 +  row.coordinates_y**2) if (row.coordinates_x < 0) else np.sqrt((-row.opponent_x - row.coordinates_x)**2 + row.coordinates_y**2), axis=1 , result_type="expand" )

data = df[["Distance", "isGoal"]]
data = data.dropna()
X = data[["Distance"]]
y = data["isGoal"].apply( lambda x : 1 if x else 0 )


experiment = Experiment( project_name="milestone-2" )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)

clf.predict_proba(X_test)

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X_train)
experiment.log_metrics(metrics)



