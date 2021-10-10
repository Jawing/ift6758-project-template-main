import numpy as np
import pandas as pd

def tidyData(dfs: pd.DataFrame) -> pd.DataFrame:
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

    Examples
    --------
    dfs = loadstats(2019,'./data/')
    df = tidyData(dfs)
    """
    rows_list, event_idx, game_id, period, periodTime, teamInfo, isGoal, shotType, \
    coordinates_x, coordinates_y, shooter, goalie, emptyNet, strength = ([] for i in range(14))

    for j in range(dfs.shape[1]): # dfs.shape[1]
        allPlays = dfs.iloc[:, j]["liveData"]["plays"]["allPlays"]
        for event in allPlays:
            # search for events that are 'shot' or 'goal'
            if event['result']['eventTypeId'] == "SHOT":
                rows_list.append(event)
                game_id.append(dfs.iloc[:, j].name)
                strength.append('NA')
            if event['result']['eventTypeId'] == "GOAL":
                rows_list.append(event)
                game_id.append(dfs.iloc[:, j].name)
                strength.append(event['result']['strength']['code'])
            # count += 1

    toCheck = [rows_list, strength, game_id]

    if len({len(i) for i in toCheck}) == 1:
        df = pd.DataFrame(rows_list)

    for i in range(df.shape[0]):

        event_idx.append(df['about'][i]['eventIdx'])
        period.append(df['about'][i]['period'])
        periodTime.append(df['about'][i]['periodTime'])
        teamInfo.append(df['team'][i]['name'])
        isGoal.append(df['result'][i]['eventTypeId'] == "GOAL")

        # coordinates_x.append(df['coordinates'][i]['x'])
        # coordinates_y.append(df['coordinates'][i]['y'])

        if 'secondaryType' in df['result'][i]:
            shotType.append(df['result'][i]['secondaryType'])
        else:
            shotType.append(pd.NA)

        if 'x' in df['coordinates'][i]:
            coordinates_x.append(df['coordinates'][i]['x'])
        else:
            coordinates_x.append(pd.NA)

        if 'y' in df['coordinates'][i]:
            coordinates_y.append(df['coordinates'][i]['y'])
        else:
            coordinates_y.append(pd.NA)


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

        if shooter_count != goalie_count:
            print("shooter_count not equal to goalie_count")
            print("i ", i)
            print("game_id", game_id[i])
            print(df['about'][i]['eventIdx'], "event idx \n")
            if not ('emptyNet' in df['result'][i] and df['result'][i]['emptyNet'] == True):
                print("score, not emptyNet, but no goalie! Add goalie as pd.NA")
                goalie.append(pd.NA)
                goalie_count += 1

        if shooter_count != goalie_count:
            raise ValueError("shooter_count not equal to goalie_count")
        # score, not emptyNet, no goalie, 2019020575 335

        # 283 event idx



        # toCheck = [game_id, rows_list, strength, event_idx, shooter, goalie]
        # it = iter(toCheck)
        # the_len = len(next(it))
        # if not all(len(l) == the_len for l in it):

        #     for lst in toCheck:
        #         print(lst, "\n")
        #     raise ValueError('not all lists have same length!')
    for lst in [event_idx, period, periodTime, teamInfo, isGoal,
                shotType, coordinates_x, coordinates_y, shooter, goalie, emptyNet, strength]:
        print(len(lst))

    df2 = pd.DataFrame(np.column_stack([game_id, event_idx, period, periodTime, teamInfo, isGoal,
                                        shotType, coordinates_x, coordinates_y, shooter, goalie, emptyNet, strength]),
                       columns=['game_id', 'event_idx', 'period', 'periodTime', 'teamInfo', 'isGoal',
                                'shotType', 'coordinates_x', 'coordinates_y', 'shooter', 'goalie', 'emptyNet',
                                'strength'])

    return df2
