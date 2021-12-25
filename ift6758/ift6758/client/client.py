# This part is a little orthogonal to the rest of the tasks, but is necessary to interface with “live” NHL games. So far, you have queried old games from bygone seasons of yore. Now, we want to put our shiny models to the test on the latest games that are happening right NOW! To do this, we need to adapt the functionality that we implemented to download game data to support a more intelligent way of pinging live games. Specifically, we want to be able to pull live game data and only process events that we haven’t yet seen. There are a few ways to do this:

# Always get the full event for the same game_id, e.g. 2021020329 would be:

# https://statsapi.web.nhl.com/api/v1/game/2021020329/feed/live/

# and get the full game back as a response. Keep some sort of internal tracker of what events you have processed so far. 
# Then, process the events that you haven’t seen (i.e. produce the features required by the model, query the prediction service, 
# and store the goal probabilities). Update the tracker you implemented so that the next time you ping the server, 
# you will be able to ignore the events that you’ve already processed. You will likely need to use the last event somewhere in order to correctly compute features for the subsequent event (as the advanced features involve including the previous event information).

# ping_game(game_id, idx, other) -> new_events, new_idx, new_other
from datetime import time
import pandas as pd
import numpy as np
import requests
import json
from ift6758.data.functions import loadstats_pergame
from ift6758.data.tidyData_single import tidyData_single
from ift6758.data.functions import pre_process
import logging
from ift6758.ift6758.client.serving_client import ServingClient

logger = logging.getLogger(__name__)
s = ServingClient()

class GameClient:
    def __init__(self, game_id = 2021020329, G_home = 0,G_away = 0, last_Idx = -1):
        #store the games processed and their last_Idx,xg,teamnames
        self.game_dic = {game_id:(G_home, G_away, last_Idx)}
        # any other potential initialization

    def ping_game(self, game_id):
        """
        Ping a game for new events.
        """
        
        last_Idx = -1
        G_homeP = 0
        G_awayP = 0
        #retrieve new events if already retrieved
        if game_id in self.game_dic:
            G_homeP, G_awayP, last_Idx = self.game_dic[game_id]


        logger.info(f"Retrieving lastest game; {game_id}")

        loadstats_pergame1 = loadstats_pergame(game_id)
        #print(len(loadstats_pergame1.iloc[:, 0]["liveData"]["plays"]["allPlays"][0:335]))
        X,last_Idx = tidyData_single(loadstats_pergame1,last_Idx)
        #no new events
        if X.empty == True:
            return X


        # print(list(X.columns))
        #print(X['homeTeam'].unique())
        #print(X['awayTeam'].unique())
        
        X_hometeam = X['homeTeam'].unique()
        X_awayteam = X['awayTeam'].unique()
        period_num = X['period']
        period_time = X['periodTime']

        X_homedf = X[X['teamInfo']==X_hometeam[0]]
        X_awaydf = X[X['teamInfo']==X_awayteam[0]]


        #preprocessing XGBoost
        X_homedf = preprocess_XGB(X_homedf)
        X_awaydf = preprocess_XGB(X_awaydf)


        #expected goal predict sum with previous goal predicts
        pred_home = s.predict(X_homedf)
        pred_away = s.predict(X_awaydf)

        print(pred_home)
        print(G_homeP)
        G_home = np.sum(pred_home.values)+G_homeP
        G_away = np.sum(pred_away.values)+G_awayP

        #store the information so far, xg, last index to game_ID
        self.game_dic[game_id] = G_home,G_away,last_Idx

        #calculate expected goal
        xG_home = G_home/(last_Idx+1)
        xG_away = G_away/(last_Idx+1)

        #print(s.predict(X))
        # pred_away.reset_index(inplace=True)
        # pred_home.reset_index(inplace=True)
        # X["xG"] = pd.concat([pred_home, pred_away], sort=False).sort_index()
        # print(pd.concat([pred_home, pred_away], sort=False).sort_index())

        return X, X_hometeam,X_awayteam,period_num,period_time,xG_home,xG_away

# from ift6758.ift6758.client.client import GameClient

# g = GameClient()
# g.ping_game(2021020329)

#preprocessing XGBoost
def preprocess_XGB(X):
    #print('tidy adva', X.shape)
    X["rebound"] = X["rebound"].apply( lambda x : 1 if x else 0 )
    X["isGoal"] = X["isGoal"].apply( lambda x : 1 if x else 0 )
    X = X[["periodSeconds","event_idx", "period", "coordinates_x", "coordinates_y","dist_goal", "angle_goal", 
                             "shotType", "eventType_last", "coordinates_x_last","coordinates_y_last", "distance_last",
                             "periodSeconds_last","rebound","angle_change","speed", "isGoal"]]
    data_xgboost_new = X.dropna()
    #print("After drop", data_xgboost_new.isna().shape)
    X_xg = data_xgboost_new.iloc[:, :-1]
    #print(".iloc", X_xg.shape)
    df = pd.get_dummies(X_xg[["shotType", "eventType_last"]])
    #print("After dummies", df.shape)
    #Concat new and the previous dataframe
    X_1 = pd.concat([data_xgboost_new,df], axis = 1)
    #print("X_1",X_1.shape)
    #dropping the two columns 
    X_new = X_1.drop(['shotType', 'eventType_last'], axis = 1)
    #print("After Dropping", X_new.shape)
    if len(X_new.columns) < 30:
                for i in range(30-len(X_new.columns)):
                        X_new['missing'+str(i)] = np.array([0 for j in range(X_new.shape[0])])
    #print('After adding missing columns, X_new', X_new.shape)

    return X_new