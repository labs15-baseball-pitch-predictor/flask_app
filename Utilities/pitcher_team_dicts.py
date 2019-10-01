from pybaseball import schedule_and_record
import psycopg2
import pandas as pd
import numpy as np
import datetime
import pickle
import time



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

    import requests


class Player:


    def __init__(self, player_name):
        input_ = "\'" + player_name +"\'"
        self.request = requests.get("http://lookup-service-prod.mlb.com/json/named.search_player_all.bam?sport_code='mlb'&active_sw='Y'&name_part=" + input_)
        try:
            self.profile = self.request.json()['search_player_all']['queryResults']['row']
        except Exception as e:
            self.profile = {'team_abbrev':'NA'}
        try:
            self.team = self.profile['team_abbrev']
        except Exception as e:
            self.team = 'NA'


def get_pitcher_team_dict(year):
    """
    Given a year
    Returns a dict of {Id #:{player_name: 'name', player_team: 'team_abbv'}
    """

    df = pd.DataFrame(query_redshift("""SELECT DISTINCT pitcher, player_name
                                    FROM pitches
                                    WHERE game_year = """ + str(year)),
                      columns = ['id', 'player_name'])

    df['player_name'] = df['player_name'].str.replace(' ', '_')
    df['player_team'] = df['player_name'].apply(lambda x: Player(x).team)
    df['player_team'] = df['player_team'].replace({'CWS':'CHW', 'WSH':'WSN'})
    df = df.set_index(['id'])

    player_teams = df.to_dict(orient = 'index')

    return player_teams


def pickle_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


for year in list(range(2010, 2020)):
    path = str(year) + '_pitcher_team_dict.pkl'
    pickle_obj(get_pitcher_team_dict(year), path)
