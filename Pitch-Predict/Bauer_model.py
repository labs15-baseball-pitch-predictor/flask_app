import pandas as pd
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, roc_auc_score


class Machine_learning:


    def __init__(self):

        # Import pitcher data
        f_myfile = open('Pitch-Predict/Bauer_df.pkl', 'rb')
        self.pitcher_df = pd.read_pickle(f_myfile, compression='zip')


    def custom_ordinal_ecode(self, df):
        df = df.copy()

        #description cols:
        desc_map = {'called_strike':1,
                    'swinging_strike':2,
                    'foul_tip':3,
                    'foul':4,
                    'swinging_strike_blocked':5,
                    'foul_bunt':6,
                    'missed_bunt':6,
                    'bunt_foul_tip':6,
                    'N/A':7,
                    'pitchout':7,
                    'hit_into_play':8,
                    'ball':9,
                    'blocked_ball':10,
                    'hit_by_pitch':11,
                    'hit_into_play_no_out':12,
                    'hit_into_play_score':13}

        desc_cols = ['L1_description', 'L2_description', 'L3_description']
        df[desc_cols] = df[desc_cols].replace(desc_map).astype('int')

        #pitch_result cols
        pitch_result_map = {'S':1, 'N/A':2, 'X':3, 'B':4}
        result_cols = ['L1_pitch_result', 'L2_pitch_result']
        df[result_cols] = df[result_cols].replace(pitch_result_map).astype('int')

        #pitch_type cols
        pitch_type_map = {'FA':1, 'FF':1, 'FT':2, 'FC':2, 'FS':2, 'SI':2, 'SF':2, 'N/A':2.5, 'SL':3,
                          'CB':4, 'CU':4, 'SC':5, 'KC':5, 'CH':6, 'KN':7, 'EP':8, 'FO':9, 'PO':9}
        pitch_type_cols = ['L1_pitch_type', 'L2_pitch_type', 'L3_pitch_type', 'pitch_type']
        df[pitch_type_cols] = df[pitch_type_cols].replace(pitch_type_map).astype('float')

        #count_cat
        count_cat_map = {'ahead':1,'neutral':2, 'behind':3}
        df['count_cat'] = df['count_cat'].replace(count_cat_map).astype('int')

        #count
        _count_map = {'02':1, '12':2, '01':3, '22':4, '11':5, '00':6, '21':7, '32':8, '10':9, '20':10, '31':11, '30':12}
        df['_count'] = df['_count'].replace(_count_map).astype('int')

        #for swung and chased, make unknown (-1) set to 0, and 0 (didnt swing/chase) set to -1:
        swung_and_chased_cols = ['L1_batter_swung', 'L1_chased', 'L2_chased', 'L3_chased']

        def swung_chase_edit(x):
            if x == 0:
                return -1
            elif x == -1:
                return 0
            else:
                return x

        for col in swung_and_chased_cols:
            df[col] = df[col].apply(swung_chase_edit)

        #fill remaining misc categories to numerics:
        misc_map = {'L':-1, 'R':2, 'Top':-1, 'Bot': 1, 'Standard':0, 'Infield shift': 1, 'Strategic':2, '4th outfielder':3}
        df = df.replace(misc_map)

        #clean up category dtypes to ints
        df['year'] = df['year'].cat.codes
        df['catcher_id'] = df['catcher_id'].cat.codes

        cat_cols = ['outs_when_up', 'inning', 'at_bat_number', 'pitch_number', 'balls', 'strikes', 'pitch_count', 'L1_pitch_zone',
                    'L1_batter_swung', 'L1_chased', 'L2_pitch_zone', 'L2_chased', 'L3_pitch_zone', 'L3_chased', 'batting_order_slot',
                    'month']

        df[cat_cols] = df[cat_cols].astype('int')
        df[['stand', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']] = df[['stand', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']].astype('int')
        return df


    def train_test_split_by_date(self, df, train_fraction):
        train_idx = int(len(df) * train_fraction)
        train_end_date = df.iloc[train_idx].game_date
        train = df[df['game_date'] < train_end_date]
        test = df[df['game_date'] >= train_end_date]
        return train, test


    def scale_numerics(self, X, X_test):
        scale_cols = ['fastball_perc_faced', 'fastball_chase_perc', 'fastball_bip_swung_perc', 'fastball_taken_strike_perc',
                  'fastball_est_woba', 'fastball_babip', 'fastball_iso_value', 'breaking_perc_faced', 'breaking_chase_perc',
                  'breaking_bip_swung_perc', 'breaking_taken_strike_perc', 'breaking_est_woba', 'breaking_babip',
                  'breaking_iso_value', 'offspeed_perc_faced', 'offspeed_chase_perc', 'offspeed_bip_swung_perc',
                  'offspeed_taken_strike_perc', 'offspeed_est_woba', 'offspeed_babip', 'offspeed_iso_value',
                  'pitchout_perc_faced', 'overall_fastball_perc', 'count_cat_fastball_perc', 'overall_breaking_perc',
                  'count_cat_breaking_perc', 'overall_offspeed_perc', 'count_cat_offspeed_perc', 'L5_fastball_perc',
                  'L15_fastball_perc', 'L5_breaking_perc', 'L15_breaking_perc', 'L5_offspeed_perc', 'L15_offspeed_perc',
                  'L5_strike_perc', 'L15_strike_perc', 'PB_fastball', 'PB_breaking', 'PB_offspeed']

        scaler = RobustScaler()
        X[scale_cols] = scaler.fit_transform(X[scale_cols].values)
        X_test[scale_cols] = scaler.transform(X_test[scale_cols].values)
        return X, X_test


    def tts(self):
        #encode cat vars
        pitcher_df = self.custom_ordinal_ecode(self.pitcher_df)

        #split into train/test
        train, test = self.train_test_split_by_date(pitcher_df, .7)

        #split into X matrix/ y vector
        target = 'pitch_type'
        drop_cols = ['player_name', 'game_date', 'pitch_cat',
                     'pitcher', target]

        X = train.drop(columns = drop_cols)
        X_test = test.drop(columns=drop_cols)

        self.y = train[target]
        self.y_test = test[target]

        #scale numerics
        self.X, self.X_test = self.scale_numerics(X, X_test)

        return self.X, self.X_test, self.y, self.y_test
