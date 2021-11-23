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
from sklearn.neural_network import MLPClassifier


# Splitting the dataset into the Training set and Test set for grid search CV 
#preprocess
#load preprocessed data
df_prep = pickle.load( open("./data/data_train_prep.pickle",'rb'))
y = df_prep['isGoal']
X = df_prep.drop(['isGoal'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)


#model pipeline
mlp = MLPClassifier(batch_size=500, activation = 'relu', learning_rate='adaptive', max_iter=200, verbose=True,  early_stopping=True, n_iter_no_change=10)

pipe = Pipeline(steps=[('scaler', StandardScaler()), ("pca", PCA(n_components = 100)), ("mlp",mlp)])

# Parameters of pipelines can be set using ‘__’ separated parameter names:
parameter_space = {
    'mlp__hidden_layer_sizes': [(60,60),(30,30),(100,100),(200)]
}


search_mlp = GridSearchCV(pipe, parameter_space, cv=3, n_jobs=-1, verbose=2)
#save model
#search_mlp.fit(X_train, y_train)
#dump(search_mlp, './models/Q6mlp_s.joblib')

#load model
search_mlp = load('./models/Q6mlp_s.joblib')

#get confusion matrix and predictions
print("Best parameter (CV score=%0.3f):" % search_mlp.best_score_)
print(search_mlp.best_params_)

y_pred = search_mlp.predict(X_test)
#print("\nResults\nConfusion matrix \n {}".format(confusion_matrix(y_test, y_pred)))
#print(confusion_matrix(y_test, y_pred))
experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y_test, y_pred))

#plot roc
from sklearn import metrics
y_score = search_mlp.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='MLP')
display.plot()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("ROC of MLP")
plt.savefig('./figures/q65_mlp_ROC.png')
plt.show()

#model percentile
lr_probs = search_mlp.predict_proba(X_test)
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
plt.title("Goal Rate of MLP")
plt.savefig('./figures/q65_mlp_GR.png')
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
plt.savefig('./figures/q65_mlp_CP.png')
plt.show()

from sklearn.calibration import CalibrationDisplay
disp = CalibrationDisplay.from_estimator(search_mlp, X_test, y_test, name='MLP')
plt.title("Reliability diagram")
plt.savefig('./figures/q65_mlp_RD.png')
plt.show()



#quantitative metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))

#these will be logged to your sklearn-demos project on Comet.ml
params={"random_state":0,
        "model_type":"mlp",
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
experiment.log_model("Q6_Full_mlp", "./models/Q6mlp_s.joblib")
#experiment.end()