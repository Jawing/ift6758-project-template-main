import os.path
import requests
import json
import pandas as pd

#function for data acquisition
def loadstats(targetyear: int, filepath: str) -> pd.DataFrame:
    """
    load NHL play-by-play data for both regular season and playoffs.
    

    Parameters
    ----------
    targetyear : int
        year season of the games.
        eg. 2016 -> 2016-2017 season
    filepath : str
        filepath = './data/'
        subgrouped in years eg. './data/targetyear/gameID.json'


    Returns
    -------
    pd.DataFrame
        pandas DataFrame of the play-by-play data for the whole year.

    Examples
    --------
    >>> loadstats(2016,'./data/')
    pd.DataFrame
    """

    #main dataframe dictionary set for all games
    data = {}
    
    #define gametype
    REGULAR_SEASON = "02"
    gameNumber = 1
    rstatus = 0
    
    #first game id
    gameIDfirst = str(targetyear) + REGULAR_SEASON + format(gameNumber, '04d')
    
    #loop through regular season
    #while the game can be found in the api and gameNumber less than or equal to 1271
    while gameNumber <= 1271 and rstatus<400:
        gameID = str(targetyear) + REGULAR_SEASON + format(gameNumber, '04d')
        filename=f'{filepath}/{targetyear}/{gameID}.json'
        #checks if dataset in targetyear exist at filepath
        if os.path.isfile(filename):
            #if exist load all data for targetyear and return as pandas Dataframe
            with open(filename) as f:
                data[gameID] = json.load(f)
                gameNumber += 1
            f.close()
            continue

        #request server api
        r = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{gameID}/feed/live/")

        #if no error at reponse, store in dataframe
        if not (r.status_code >= 400):
            #check for different status code other than 200
            if r.status_code != 200:
                print(f'Status code: {r.status_code} at gameID:{gameID}. Unexpected.')
            #save and store in data folder
            data[gameID] = r.json()
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data[gameID], f, ensure_ascii=False, indent=4)
            f.close()
            gameNumber += 1
            continue
        else:
            #game not added if it does not exist
            print(f'Error code: {r.status_code} at gameID:{gameID}. Game not found.')
            rstatus = r.status_code
            gameNumber += 1
    print(f'size of data in regular season: {len(data)}')
    
    #store the index where the playoffgames begin in the metaData section of first game
    data[gameIDfirst]['metaData']['playoffIndex'] = len(data)
    
    #4 rounds, first round 8 faceoffs, second round 4 faceoffs, third round 2 faceoffs. final round out of 7
    PLAYOFFS= "03"

    playoffround = 1 
    #note 0 round (qualifying) in 2019-2020
    if targetyear == 2019:
        playoffround = 0
        

    #loop through 4 playoffs rounds
    while playoffround <= 4:
        matchup = 1
        if playoffround == 0:
            matchup = 0
        #define number of matchups in each round
        if (playoffround == 0):
            #9 match ups in 2019
            matchupmax = 9
        elif (playoffround == 1):
            matchupmax = 8
        elif (playoffround == 2):
            matchupmax = 4
        elif (playoffround == 3):
            matchupmax = 2
        elif (playoffround == 4):
            matchupmax = 1
        #loop through matchups
        while matchup <= matchupmax:
            rstatus = 0
            playoffgame = 1
            #loop through games up to 7
            while playoffgame <= 7 and rstatus<400:
                gameID = str(targetyear) + PLAYOFFS + '0' + str(playoffround) + str(matchup) + str(playoffgame)
                filename=f'{filepath}/{targetyear}/{gameID}.json'
                #checks if dataset in targetyear exist at filepath
                if os.path.isfile(filename):
                    #if exist load all data for targetyear and return as pandas Dataframe
                    with open(filename) as f:
                        data[gameID] = json.load(f)
                        playoffgame += 1
                    f.close()
                    continue

                #request server api
                r = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{gameID}/feed/live/")

                #if no error at reponse, store in dataframe
                if not (r.status_code >= 400):
                    #check for different status code other than 200
                    if r.status_code != 200:
                        print(f'Status code: {r.status_code} at gameID:{gameID}. Unexpected.')
                    #save and store in data folder
                    data[gameID] = r.json()
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'w') as f:
                        json.dump(data[gameID], f, ensure_ascii=False, indent=4)
                    f.close()
                    playoffgame += 1
                    continue
                else:
                    #game not added if it does not exist
                    print(f'Error code: {r.status_code} at gameID:{gameID}. Game not found.')
                    rstatus = r.status_code
                    playoffgame += 1
            matchup += 1
        playoffround += 1
    print(f'size of data in regular season & playoffs: {len(data)}')
    

    return pd.DataFrame.from_dict(data)


def getNHLData():
    """
        function to convert all events of every game into a pandas dataframe.
        Use your tool to download data from the 2016-17 season all the way up to the 2020-21 season. 

    """
    # get all games for the requsted period by calling above funciion
    # keep adding to a pandas data frame
    # Columns = [ game time/period information, game ID, team information (which team took the shot), indicator if its a shot or a goal, the on-ice coordinates, the shooter and goalie name (don’t worry about assists for now), shot type, if it was on an empty net, and whether or not a goal was at even strength, shorthanded, or on the power play ]
    for season in list(range(2016, 2021)):
        print("Loading data for {season}", season)
        loadstats(season, './data')

        
