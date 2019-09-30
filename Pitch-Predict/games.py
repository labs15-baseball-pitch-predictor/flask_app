from .column_names import col_names

import psycopg2
import pandas as pd
import requests
from pybaseball import schedule_and_record
import numpy as np
import datetime



class Player:


  def __init__(self, player_name):
    input_ = "\'" + player_name +"\'"
    self.request = requests.get(
    """http://lookup-service-prod.mlb.com/json/named.search_player_all.
       bam?sport_code='mlb'&active_sw='Y'&name_part=""" + input_
    )
    try:
        self.profile = self.request.json()['search_player_all']['queryResults']['row']
    except Exception as e:
        self.profile = {'team_abbrev':'NA'}
    try:
        self.team = self.profile['team_abbrev']
    except Exception as e:
        self.team = 'NA'



def query_redshift(query_string):

    con = psycopg2.connect(
        dbname= 'dev',
        host='examplecluster.cdbpwaymevt5.us-east-2.redshift.amazonaws.com',
        port= '5439',
        user= 'awsuser',
        password= '47Westrange')

    cur = con.cursor()

    cur.execute(query_string)

    results = cur.fetchall()

    cur.close()
    con.close()

    return results


def today():

    df = pd.DataFrame(query_redshift("""SELECT DISTINCT pitcher, player_name
                                        FROM pitches
                                        WHERE game_year = 2019"""),
                                        columns = ['id', 'player_name'])

    df['player_name'] = df['player_name'].str.replace(' ', '_')

    df['player_team'] = df['player_name'].apply(lambda x: Player(x).team)


    today = pd.to_datetime(datetime.date.today())

    teams = ['OAK', 'LAD', 'TOR', 'PHI', 'ATL', 'LAA', 'BAL', 'HOU', 'BOS',
             'CIN', 'SD', 'TEX', 'PIT', 'COL', 'STL', 'CHW', 'CHC', 'TB',
             'MIN', 'DET', 'ARI', 'SF', 'KC', 'WSN', 'SEA', 'MIA', 'NYY',
             'MIL', 'CLE', 'NYM']

    todays_games = []

    for team in teams:
        data = schedule_and_record(2019, team)
        data['year'] = np.ones(data.shape[0], int) * 2019

        data['month'] = data['Date'].str.split(' ').apply(lambda x: x[1])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']
        month_mapping = dict(zip(months, list(range(1,13))))
        data['month'] = data['month'].map(month_mapping)

        data['day'] = data['Date'].str.split(' ').apply(lambda x: x[2])

        data['Date'] = (data['year'].astype(str) + ', ' +
                        data['month'].astype(str) +  ', ' +
                        data['day'].astype(str))
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[['Date', 'Tm', 'Home_Away', 'Opp']]

        todays_games.append(data[data['Date'] == today].to_dict())

        games_on_date = []

    for i in todays_games:
        team = ''.join(list(i['Tm'].values()))
        home_or_away = ''.join(list(i['Home_Away'].values()))
        opposing_team = ''.join(list(i['Opp'].values()))
        if home_or_away == '@':
            games_on_date.append(
                str(str(team) + ' ' +
                    str(home_or_away) + ' ' +
                    str(opposing_team))
                )

    today = pd.DataFrame(games_on_date, columns = ['todays_games'])

    today['home_team'] = today['todays_games'].str.split().apply(lambda x : x[2])
    today['away_team'] = today['todays_games'].str.split().apply(lambda x : x[0])

    def team_pitchers(team):
        """Takes a list of teams and returns a list of pitcher ids"""
        team_pitchers = df.where(df['player_team'] == team).dropna(axis = 0)
        team_pitchers = team_pitchers['id'].tolist()
        return team_pitchers

    def team_pitchers_names(team):
        """Takes a list of teams and returns a list of pitcher ids"""
        team_pitchers = df.where(df['player_team'] == team).dropna(axis = 0)
        team_pitchers = team_pitchers['player_name'].tolist()
        return team_pitchers

    today['home_pitchers'] = today['home_team'].apply(lambda x: team_pitchers(x))
    today['away_pitchers'] = today['away_team'].apply(lambda x: team_pitchers(x))
    today['home_pitchers_names'] = today['home_team'].apply(lambda x: team_pitchers_names(x))
    today['away_pitchers_names'] = today['away_team'].apply(lambda x: team_pitchers_names(x))

    return today
