#
#   Test on 2019 data with model from registry
#
#


#import model tracking
from comet_ml import init, API, Experiment


import pickle
from xgboost import XGBClassifier  

# Init comet ml
init()

# Start an experiment
experiment = Experiment(
    project_name="milestone-2",
    workspace="binulal",
)

# Get data and pre process
data_xgboost = pickle.load(open('data/data_test_rg_tidy.pickle', 'rb'));

#changing categorical 
data_xgboost["rebound"] = data_xgboost["rebound"].apply( lambda x : 1 if x else 0 )

#print(data_xgboost.isna().sum())
features = ["periodSeconds", "period", "coordinates_x", "coordinates_y",
            "dist_goal", "angle_goal", "shotType", "eventType_last",
            "coordinates_x_last","coordinates_y_last", "distance_last",
            "periodSeconds_last","rebound","angle_change","speed" ]
target = ["isGoal"]

data_xgboost = data_xgboost.dropna()

X = data_xgboost[ features ]
y = data_xgboost[ target ]

# Initialize API
api = API()

# Download a Registry Model
api.download_registry_model( "binulal", "q5-xgboost-tuned", "1.0.0", output_path="./models/", expand=True )

xgb_clf = pickle.load(open('./models/Q52_XGboost_hyperparameter.pkl', 'rb'))

y_pred = xgb_clf.predict(X)

experiment.log_confusion_matrix(labels=["No_Goal", "Goal"],
  matrix=confusion_matrix(y, y_pred))


#quantitative metrics
f1 = f1_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))


metrics = {
    "f1" : f1,
    "recall" : recall,
    "precision" : precision
}

experiment.log_dataset_hash(X)
experiment.log_metrics(metrics)

