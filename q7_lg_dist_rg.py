#
#   Test Lg distance model using 2019 data
#
#

#import model tracking
from comet_ml import init, API, Experiment

# Init comet ml
init()

# Start an experiment
experiment = Experiment(
    project_name="milestone-2",
    workspace="binulal",
)

#imports
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from ift6758.data.functions import loadstats
from ift6758.data.functions import pre_process
from ift6758.data.tidyData_adv import tidyData_adv

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#import plotly.express as px
#import pandas as pd
#import numpy as np
#import json
import pickle


# Get test data set

#dfs_2019_po = loadstats(2019,'./data/', playoffs=True,regular=False)
#df_test_po = tidyData_adv(dfs_2019_po)

#dfs_2019_rg = loadstats(2019,'./data/', playoffs=False,regular=True)
#df_test_rg = tidyData_adv(dfs_2019_rg)


#df_all = df_test_po.append(df_test_rg, ignore_index=True)
#df_test = pre_process(df_test_rg)

# Initialize API
api = API()

# Download a Registry Model
api.download_registry_model( "binulal", "q3-logreg-distance", "2.0.0", output_path="./models/", expand=True )

df_test = pickle.load(open('data/data_test_po_tidy.pickle', 'rb'));

data_distance = df_test[["dist_goal", "isGoal"]].dropna()

X = data_distance[["dist_goal"]]
y = data_distance["isGoal"].apply( lambda x : 1 if x else 0 )

print( X, y )

#load model pickle.load(open('model.pkl', 'rb'))
clf = pickle.load(open('./models/Q31_log_reg_distance.pkl', 'rb'))

y_pred = clf.predict(X)

print("Predictions : ", y_pred)

experiment.log_confusion_matrix(
    labels=["No_Goal", "Goal"],  matrix=confusion_matrix(y, y_pred) )


#quantitative metrics
f1 = f1_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))


metrics = {"f1":f1,
"recall":recall,
"precision":precision
}

experiment.log_dataset_hash(X)
experiment.log_metrics(metrics)
