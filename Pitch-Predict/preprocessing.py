import category_encoders
from category_encoders.one_hot import OneHotEncoder

class Preprocess:


    def __init__(self, dataframe):
        self.df = dataframe


    def initial_column_drop(self, dataframe):
        dr_columns = [
            'release_pos_x', 'release_pos_z',
            #might want to keep bb_type and filder_2 is the catcher
            'hit_location', 'bb_type', 'hc_x', 'hc_y', 'fielder_2',
            'hit_distance_sc', 'launch_speed', 'launch_angle',
            'effective_speed', 'release_spin_rate', 'release_extension',
            'release_pos_y', 'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle', 'woba_value','woba_denom',
            'babip_value', 'iso_value', 'launch_speed_angle',
            'if_fielding_alignment', 'of_fielding_alignment', 'sv_id', 'pfx_x',
            'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
            'az', 'sz_top', 'sz_bot', 'post_away_score', 'post_home_score',
            'post_bat_score', 'post_fld_score', 'pitch_name', 'pitcher_1',
            'fielder_2_1',	'fielder_3',	'fielder_4',	'fielder_5',	'fielder_6',
            	'fielder_7',	'fielder_8',	'fielder_9', 'level_0'
            ]

        drop_columns = []
        for col in dr_columns:
            if col in dataframe.columns.tolist():
                drop_columns.append(col)

        dataframe = dataframe.drop(columns = drop_columns)

        dataframe = dataframe[dataframe.pitch_type != 'pitchout']

        return dataframe


    def data_wrangle(self, dataframe):

        """
        ## Events
        **Feature Description**: Event of the resulting Plate Appearance.
        **Issue**: Feature is categorical, and polluted with null values
        **Solution**: Impute null values with placeholder, then one-hot encode
                      the feature while dropping the resulting placeholder
                      column
        """
        dataframe['events'] = dataframe['events'].fillna(value = 0)

        """
        ## Release Speed
        **Feature Description**: Pitch velocities from 2008-16 are via
                                 Pitch F/X, and adjusted to roughly out-of-hand
                                 release point. All velocities from 2017 and
                                 beyond are Statcast, which are reported
                                 out-of-hand.
        **Issue**: Continuous numerical feature with 109 null values
        **Solution**: Impute null values median release speed
        """
        speed_median = dataframe['release_speed'].median()
        dataframe['release_speed'] = dataframe['release_speed'].fillna(value = speed_median)

        """
        ## Zone
        **Feature Description**: Zone location of the ball when it crosses the
                                 plate from the catcher's perspective.
        **Issue**: Discrete numerical feature with 112 null values
        **Solution**: TEMPORARY - fill with center zone, then one-hot encode
        """
        dataframe['zone'] = dataframe['zone'].fillna(value = 5)

        """
        ## On 1B ID
        **Feature Description**: ID number of batter on fist base
        **Issue**: Categorical feature, polluted with 4863 null values
        **Solution**: TEMPORARY - Drop feature
        """
        # dataframe = dataframe.drop(columns = ['on_1b_id'])
        return dataframe


    def create_target_feature(self, dataframe):
        """
        ## Create Next Pitch Feature
        """
        dataframe['next_pitch'] = dataframe['pitch_type'].shift(-1)
        index_pos = dataframe.shape[0] - 1
        dataframe.at[index_pos, 'next_pitch'] = 'fastball'
        return dataframe


    def one_hot_encode(self, dataframe):
        one_cols = [
            'events', 'zone', 'pitch_type', 'type', 'home_team', 'away_team',
            'pitch_count', 'L1_pitch_type',	'L1_pitch_result', 'L1_pitch_zone',
            '_count', 'count_cat',	'pitch_cat', 'balls', 'strikes','inning',
            'outs_when_up', 'batting_order_slot', 'pitch_subtype', 'count_status'

            ]

        one_hot_cols = []
        for col in one_cols:
            if col in dataframe.columns.tolist():
                one_hot_cols.append(col)


        # Instantiate Encoder
        one_hot_encoder = OneHotEncoder(cols=one_hot_cols,
                                        return_df=True,
                                        use_cat_names=True)
        # Encode features
        encoded = one_hot_encoder.fit_transform(dataframe[one_hot_cols],
                                                dataframe['next_pitch'])

        # Join encoded features into df and drop old columns
        dataframe = dataframe.join(encoded).drop(columns = one_hot_cols)
        return dataframe


    def binary_encode(self, dataframe):
        """
        ## Binary Encode
        """
        dataframe['inning_topbot'] = dataframe['inning_topbot'].replace({'Top':0,
                                                                     'Bot':1})
        dataframe['stand'] = dataframe['stand'].replace({'L':0, 'R':1})
        dataframe['p_throws'] = dataframe['p_throws'].replace({'L':0, 'R':1})

        tf_map = {'True': 1, 'False': 0}

        dataframe['ball_high'] = dataframe['ball_high'].map(tf_map)
        dataframe['ball_low'] = dataframe['ball_low'].map(tf_map)
        dataframe['ball_left'] = dataframe['ball_left'].map(tf_map)
        dataframe['ball_right'] = dataframe['ball_right'].map(tf_map)

        return dataframe


    def secondary_column_drop(self, dataframe):
        """
        ## Drop Unneeded Columns
        """
        drop_cols = [
            'game_date', 'index', 'pitcher', 'batter', 'game_year', 'game_pk',
            'player_name', 'description'
            ]

        dataframe = dataframe.drop(columns = drop_cols)
        return dataframe


    def process(self, dataframe):
        df = self.initial_column_drop(dataframe)
        df = self.data_wrangle(df)
        df = self.create_target_feature(df)
        df = self.one_hot_encode(df)
        df = self.binary_encode(df)
        df = self.secondary_column_drop(df)
        return df


def main(dataframe):
    obj = Preprocess(dataframe)
    return Preprocess.process(obj.df)


if __name__ == "__main__":
    main()
