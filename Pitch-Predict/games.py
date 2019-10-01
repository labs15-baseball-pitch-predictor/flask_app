from pybaseball import schedule_and_record
import pandas as pd
import numpy as np
import datetime
import pickle



def get_games(year, month, day):
    """
    Date format XXXX, X, X
    returns a list of game matchups as strings
    """
    date = pd.to_datetime(datetime.date(year, month, day))

    teams = ['OAK', 'LAD', 'TOR', 'PHI', 'ATL', 'LAA', 'BAL', 'HOU', 'BOS',
          'CIN', 'SD', 'TEX', 'PIT', 'COL', 'STL', 'CHW', 'CHC', 'TB',
          'MIN', 'DET', 'ARI', 'SF', 'KC', 'WSN', 'SEA', 'MIA', 'NYY',
          'MIL', 'CLE', 'NYM']

    dates_games = []

    for team in teams:
        data = schedule_and_record(year, team)
        data['year'] = np.ones(data.shape[0], int) * year

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

        dates_games.append(data[data['Date'] == date].to_dict())

    games_on_date = []

    for i in dates_games:
        team = ''.join(list(i['Tm'].values()))
        home_or_away = ''.join(list(i['Home_Away'].values()))
        opposing_team = ''.join(list(i['Opp'].values()))
        if home_or_away == '@':
            games_on_date.append(
                str(str(team) + ' ' +
                    str(home_or_away) + ' ' +
                    str(opposing_team)))

    return games_on_date


def get_teams(list_of_games):
    """takes a list of game matchups
    returns a list of teams"""
    teams = []
    for i in list_of_games:
        away_team = i.split()[0]
        home_team = i.split()[2]
        teams.append(home_team)
        teams.append(away_team)

    return teams


def get_teams_pitchers(team, year):
    """Returns a list of player ids for a specified team abbrivation and year"""

    path = "Pitch-Predict/pitcher_team_dicts/" + str(year) + '_pitcher_team_dict.pkl'
    pitcher_team_dict = pickle.load(open(path, 'rb'))

    pitchers = []
    for id, player_info in pitcher_team_dict.items():
        if player_info['player_team'] == team:
            pitchers.append(id)

    return pitchers


def get_all_pitchers(year, month, day):
    """
    takes a date xxxx, x, x
    returns a list of lists containing pitcher_ids
    """
    all_pitchers = []
    for team in get_teams(get_games(2019, 6, 2)):
        all_pitchers.append(get_teams_pitchers(team, year))

    all_pitchers = [item for sublist in all_pitchers for item in sublist]

    return all_pitchers
