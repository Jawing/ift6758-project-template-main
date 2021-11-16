import numpy as np
import pandas as pd
import datetime
import time
from ift6758.data.functions import distAngle_FromGoal

def tidyData_adv(dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy the json-df downloaded with loadstats() and return a tidy df ready to use..

    Parameters
    ----------
    dfs : pd.DataFrame
        json-df downloaded with loadstats()

    Returns
    -------
    pd.DataFrame
        pandas DataFrame of the play-by-play data where each row is an play event.

        "For this milestone, you will want to include events of the type “shots” and “goals”.
        You can ignore missed shots or blocked shots for now.
        For each event, you will want to include as features (at minimum):
        game time/period information - period, periodTime
        game ID - game_id
        team information (which team took the shot) - teamInfo
        indicator if its a shot or a goal - isGoal
        the on-ice coordinates - coordinates_x, coordinates_y
        the shooter and goalie name (don’t worry about assists for now) - shooter, goalie
        shot type - shotType
        if it was on an empty net - emptyNet
        and whether or not a goal was at even strength,
        shorthanded, or on the power play."
        homeTeam,awayTeam names
        homeSide starting side
    Examples
    --------
    dfs = loadstats(2019,'./data/')
    df = tidyData(dfs)
    """
    #define list for each feature
    rows_list, last_event, speed, periodSeconds_last, eventType_last, rebound, event_idx, game_id, period, periodType, periodTime,periodSeconds, teamInfo, isGoal, shotType, \
    coordinates_x, coordinates_y, coordinates_x_last, coordinates_y_last, dist_goal, angle_goal, angle_change,distance_last, angle_speed, shooter, goalie, emptyNet, strength,homeTeam,awayTeam, homeSide = ([] for i in range(31))

    #loop through all games in the year
    for j in range(dfs.shape[1]): # dfs.shape[1]
        allPlays = dfs.iloc[:, j]["liveData"]["plays"]["allPlays"]
        
        #define last event
        last_event_cache = pd.NA

        #loop through allplays in the game
        for event in allPlays:
            # search for events that are 'shot' or 'goal'
            if event['result']['eventTypeId'] == "SHOT":
                last_event.append(last_event_cache)
                rows_list.append(event)
                game_id.append(dfs.iloc[:, j].name)
                strength.append('NA')
                awayTeam.append(dfs.iloc[:, j]['gameData']['teams']['away']['name'])
                homeTeam.append(dfs.iloc[:, j]['gameData']['teams']['home']['name'])
                #check for rinkSide info and append
                try:
                    homeSide.append(dfs.iloc[:, j]['liveData']['linescore']['periods'][0]['home']['rinkSide'])
                except:
                    homeSide.append('NA')
            if event['result']['eventTypeId'] == "GOAL":
                last_event.append(last_event_cache)
                rows_list.append(event)
                game_id.append(dfs.iloc[:, j].name)
                strength.append(event['result']['strength']['code'])
                awayTeam.append(dfs.iloc[:, j]['gameData']['teams']['away']['name'])
                homeTeam.append(dfs.iloc[:, j]['gameData']['teams']['home']['name'])
                try:
                    homeSide.append(dfs.iloc[:, j]['liveData']['linescore']['periods'][0]['home']['rinkSide'])
                except:
                    homeSide.append('NA')
            last_event_cache = event
            # count += 1

    toCheck = [last_event, rows_list, strength, game_id,awayTeam, homeTeam,homeSide]
    
    #check if size of each list is equal
    if len({len(i) for i in toCheck}) == 1:
        df = pd.DataFrame(rows_list)
        df_last = pd.DataFrame(last_event)
    #print(df)
    
    #loop through each event data
    for i in range(df.shape[0]):

        #event period info
        event_idx.append(df['about'][i]['eventIdx'])
        period.append(df['about'][i]['period'])
        periodType.append(df['about'][i]['periodType'])
        periodTime.append(df['about'][i]['periodTime'])
        teamInfo.append(df['team'][i]['name'])
        isGoal.append(df['result'][i]['eventTypeId'] == "GOAL")

        #period time in seconds
        date_time = datetime.datetime.strptime(df['about'][i]['periodTime'], "%M:%S")
        a_timedelta = date_time - datetime.datetime(1900, 1, 1)
        periodSeconds.append(a_timedelta.total_seconds())
        
        #period time from last event
        date_time_last = datetime.datetime.strptime(df_last['about'][i]['periodTime'], "%M:%S")
        a_timedelta_last = date_time - date_time_last
        periodSeconds_last.append(a_timedelta_last.total_seconds())

        #last event type
        eventType_last.append(df_last['result'][i]['eventTypeId'])


        # coordinates_x.append(df['coordinates'][i]['x'])
        # coordinates_y.append(df['coordinates'][i]['y'])

        if 'secondaryType' in df['result'][i]:
            shotType.append(df['result'][i]['secondaryType'])
        else:
            shotType.append(pd.NA)

        na_coor = False
        na_coor_c = False
        #get coordinates from event
        if 'x' in df['coordinates'][i]:
            c_x = df['coordinates'][i]['x']
            coordinates_x.append(c_x)
        else:
            na_coor_c = True
            na_coor = True
            coordinates_x.append(pd.NA)
        if 'y' in df['coordinates'][i]:
            c_y = df['coordinates'][i]['y']
            coordinates_y.append(c_y)
        else:
            na_coor_c = True
            na_coor = True
            coordinates_y.append(pd.NA)

        if na_coor_c == False:
            dist_g,angle_g = distAngle_FromGoal(c_x,c_y,homeSide[i],period[i],teamInfo[i],homeTeam[i],awayTeam[i],periodType[i])
            dist_goal.append(dist_g)
            angle_goal.append(angle_g)
        else:
            dist_goal.append(pd.NA)
            angle_goal.append(pd.NA)

        #get coordinates from last event
        if 'x' in df_last['coordinates'][i]:
            l_x = df_last['coordinates'][i]['x']
            coordinates_x_last.append(l_x)
        else:
            na_coor = True
            coordinates_x_last.append(pd.NA)
        if 'y' in df_last['coordinates'][i]:
            l_y = df_last['coordinates'][i]['y']
            coordinates_y_last.append(l_y)
        else:
            na_coor = True
            coordinates_y_last.append(pd.NA)

        #calculate distance between current and last event if both coordinate exist
        if na_coor == False:
            dist_last = np.sqrt((l_x - c_x)**2 + (l_y-c_y)**2)
            distance_last.append(dist_last)
            speed.append(dist_last/a_timedelta_last.total_seconds())
            
        else:
            distance_last.append(pd.NA)
            speed.append(pd.NA)
        


        #rebound from last event, given last SHOT from same team and period.
        if (df_last['result'][i]['eventTypeId'] == "SHOT") and (df['about'][i]['period'] == df_last['about'][i]['period']) and (df['team'][i]['name'] == df_last['team'][i]['name']): 
            rebound.append(True)
            if len(angle_goal)>=2 and pd.notnull(angle_goal[-1]) and pd.notnull(angle_goal[-2]):
                angle_c = abs(angle_goal[-2]-angle_goal[-1])
                if angle_c > 180:
                    angle_change.append(360-angle_c)
                else:
                    angle_change.append(angle_c)
                #Change in shot angle after rebound TODO
                angle_speed.append(angle_c/a_timedelta_last.total_seconds())
            else:
                angle_change.append(0)
                angle_speed.append(pd.NA)
        else:
            rebound.append(False)
            angle_change.append(0)
            angle_speed.append(pd.NA)



        #shooter/goalie and emptynet information
        shooter_count = 0
        goalie_count = 0
        for player_info in df['players'][i]:

            if player_info['playerType'] == 'Scorer' or player_info['playerType'] == 'Shooter':
                shooter.append(player_info['player']['fullName'])
                shooter_count += 1

            if player_info['playerType'] == 'Goalie':
                goalie.append(player_info['player']['fullName'])
                goalie_count += 1

        if 'emptyNet' in df['result'][i] and df['result'][i]['emptyNet'] == True:
            emptyNet.append(True)
            goalie.append("EmptyNet") # When there is no goalie, return "EmptyNet", or maybe pd.NA?
            goalie_count += 1
            # print("emptyNet = True")
            # print(" i ", i)
            # print(" game_id", game_id[i])
            # print(df['about'][i]['eventIdx'], "event idx \n")
        else:
            emptyNet.append(False)

        if not ('emptyNet' in df['result'][i] and df['result'][i]['emptyNet'] == True) and (shooter_count > 0 and goalie_count == 0):
            # print("shooter_count not equal to goalie_count")
            # print("i ", i)
            # print("game_id", game_id[i])
            # print(df['about'][i]['eventIdx'], "event idx \n")
            # print("score, not emptyNet, but no goalie! Add goalie as pd.NA")
            goalie.append(pd.NA)
            goalie_count += 1
        if shooter_count != goalie_count:
            raise ValueError("shooter_count not equal to goalie_count")




    #shorthand check if all lens are equal
    assert(all(len(lst) == len(event_idx) for lst in [coordinates_y, coordinates_x_last, coordinates_y_last, distance_last,dist_goal, angle_goal, angle_change, angle_speed, shooter, goalie, emptyNet, strength, homeTeam,awayTeam, homeSide]) )



    assert(all(len(lst) == len(event_idx) for lst in [event_idx, last_event, speed, periodSeconds_last, eventType_last, rebound, event_idx, game_id, period, periodType, periodTime,periodSeconds, teamInfo, isGoal, shotType, coordinates_x]) )


    

    df2 = pd.DataFrame(np.column_stack([game_id, event_idx, speed, periodSeconds_last, eventType_last, rebound, period, periodType, periodTime,periodSeconds, teamInfo, isGoal, shotType, coordinates_x, coordinates_y, coordinates_x_last, coordinates_y_last, distance_last,dist_goal, angle_goal, angle_change, angle_speed, shooter, goalie, emptyNet, strength,homeTeam,awayTeam, homeSide]),
                       columns=['game_id', 'event_idx', 'speed', 'periodSeconds_last', 'eventType_last', 'rebound', 'period', 'periodType', 'periodTime','periodSeconds', 'teamInfo', 'isGoal', 'shotType', 'coordinates_x', 'coordinates_y', 'coordinates_x_last', 'coordinates_y_last', 'distance_last','dist_goal', 'angle_goal', 'angle_change', 'angle_speed', 'shooter', 'goalie', 'emptyNet', 'strength','homeTeam','awayTeam', 'homeSide'])

    return df2
