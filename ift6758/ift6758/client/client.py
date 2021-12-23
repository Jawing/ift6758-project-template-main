# This part is a little orthogonal to the rest of the tasks, but is necessary to interface with “live” NHL games. So far, you have queried old games from bygone seasons of yore. Now, we want to put our shiny models to the test on the latest games that are happening right NOW! To do this, we need to adapt the functionality that we implemented to download game data to support a more intelligent way of pinging live games. Specifically, we want to be able to pull live game data and only process events that we haven’t yet seen. There are a few ways to do this:

# Always get the full event for the same game_id, e.g. 2021020329 would be:

# https://statsapi.web.nhl.com/api/v1/game/2021020329/feed/live/

# and get the full game back as a response. Keep some sort of internal tracker of what events you have processed so far. 
# Then, process the events that you haven’t seen (i.e. produce the features required by the model, query the prediction service, 
# and store the goal probabilities). Update the tracker you implemented so that the next time you ping the server, 
# you will be able to ignore the events that you’ve already processed. You will likely need to use the last event somewhere in order to correctly compute features for the subsequent event (as the advanced features involve including the previous event information).

# ping_game(game_id, idx, other) -> new_events, new_idx, new_other
import pandas as pd
import numpy as np
import requests
import json
from ift6758.data.functions import loadstats_pergame
from ift6758.data.tidyData import tidyData
from ift6758.data.tidyData_adv import tidyData_adv
from ift6758.data.functions import pre_process
#from ift6758.client.serving_client import ServingClient

def ping_game(game_id, idx):
    """
    Ping a game for new events.
    """
    #client = ServingClient()
    loadstats_pergame1 = loadstats_pergame(game_id)
    X = tidyData_adv(loadstats_pergame1)
    print(list(X.columns))

    print(X['homeTeam'].unique())
    print(X['awayTeam'].unique())

    X_hometeam = X['homeTeam'].unique()
    X_awayteam = X['awayTeam'].unique()


    print('tidy adva', X.shape)

    X["rebound"] = X["rebound"].apply( lambda x : 1 if x else 0 )
    X["isGoal"] = X["isGoal"].apply( lambda x : 1 if x else 0 )

    X = X[["periodSeconds","event_idx", "period", "coordinates_x", "coordinates_y","dist_goal", "angle_goal", 
                             "shotType", "eventType_last", "coordinates_x_last","coordinates_y_last", "distance_last",
                             "periodSeconds_last","rebound","angle_change","speed", "isGoal"]]


    data_xgboost_new = X.dropna()
    print("After drop", data_xgboost_new.isna().shape)

    X_xg = data_xgboost_new.iloc[:, :-1]

    print(".iloc", X_xg.shape)

    df = pd.get_dummies(X_xg[["shotType", "eventType_last"]])

    print("After dummies", df.shape)

    #Concat new and the previous dataframe
    X_1 = pd.concat([data_xgboost_new,df], axis = 1)

    print("X_1",X_1.shape)

    #dropping the two columns 
    X_new = X_1.drop(['shotType', 'eventType_last'], axis = 1)
    print("After Dropping", X_new.shape)

    if len(X_new.columns) < 30:
                for i in range(30-len(X_new.columns)):
                        X_new['missing'+str(i)] = np.array([0 for j in range(X_new.shape[0])])
    print('After adding missing columns, X_new', X_new.shape)

    if X_new['event_idx'].iloc[0] == idx:
        other = X_new.iloc[0]
        print("No new events")
        return None, idx, other
    else:   
        print("New events")
        new_events = X_new.loc[X_new['event_idx'] > idx]
        new_idx = new_events['event_idx'].iloc[0]
        new_other = new_events.drop(['event_idx'], axis=1)
        return new_events, new_idx, new_other
    #return df


ping_game('2017021065', 10)

    

    
    