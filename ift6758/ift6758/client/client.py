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
from ift6758.data.tidyData_adv import tidyData_adv

def ping_game(game_id, idx):
    """
    Ping a game for new events.
    """
    loadstats_pergame1 = loadstats_pergame(game_id)
    #loadstats_pergame1 = loadstats_pergame1.drop(['isGoal'], axis=1)
    df_gameid = tidyData_adv(loadstats_pergame1)
    #print(df_gameid)
    
    if df_gameid['event_idx'].iloc[0] == idx:
        other = df_gameid.iloc[0]
        print("No new events")
        return None, idx, other
    else:   
        print("New events")
        new_events = df_gameid.loc[df_gameid['event_idx'] > idx]
        new_idx = new_events['event_idx'].iloc[0]
        new_other = new_events.drop(['event_idx'], axis=1)
        return new_events, new_idx, new_other


# ping_game('2017021065', 10)
    

    
    