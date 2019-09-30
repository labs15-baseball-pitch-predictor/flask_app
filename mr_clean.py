import pandas as pd
import numpy as np
import random
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



class Season_cleaner:
    """
    Cleans a season dataframe
    """


    def __init__(self, dataframe):

        self.df = dataframe

        # Features to drop
        self.drop_columns = [
            'spin_dir',
            'spin_rate_deprecated',
            'break_angle_deprecated',
            'break_length_deprecated',
            'game_type',
            'tfs_deprecated',
            'tfs_zulu_deprecated',
            'umpire'
        ]

        # List fo unique pitcher ID's
        self.pitchers = self.df['pitcher'].unique().tolist()


    def drop_features(self):
        """
        Drops depriciated features
        """
        self.df = self.df.drop(columns = self.drop_columns)


    def drop_instances(self):
        """
        Drops useless instances
        """
        self.df = self.df.dropna(axis = 0, how = 'all')


    def fielding_alignment_typecast(self):
        """
        Forces the object type onto the fielding allignment columns
        """
        self.df['if_fielding_alignment'] = self.df['if_fielding_alignment'].astype(object)
        self.df['of_fielding_alignment'] = self.df['of_fielding_alignment'].astype(object)


    def chronological_sort(self):
        """
        Sort pitches chronologically
        """
        self.df = self.df.sort_values(by = [
            'game_date',
            'game_pk',
            'at_bat_number',
            'pitch_number'
            ])


    def pitch_type(self):
        """
        Feature Name: pitch_type
        Feature Description: The type of pitch derived from Statcast.
        Issue: Feature is supposed to contain a 2 character string, but many values (265) are filled with long strings of numerical characters. Example: 160421_181540
        Solution: Replace values longer than 2 characters in lengeth with np.NaN
        """

        self.df['pitch_type'] = self.df.apply(
            lambda row: np.NaN\
                if len(str(row['pitch_type'])) > 2\
                else row['pitch_type'], axis = 1)

        """
        Issue: Many values of this feature are recorded as 'UN'
        Solution: Replace value with np.NaN
        """
        self.df['pitch_type'] = self.df['pitch_type'].replace({'UN':np.nan})

        """
        Issue**: The pitch type feature is filled with NaN values
        Solution: We will create a mapping of a pitchers id and his normalized pitch counts. Using these normalized values as weights we will select a random pitch type and fill the NaN value for that pitcher. We will use df.apply, but this could be time optomized by using series vectorization.
        """

        # Populate mapping
        pitcher_dict = {}
        for pitcher in self.pitchers:

            # Pitcher's prior pitch type probabilites
            pitch_type_weights = self.df[self.df.pitcher == pitcher]\
                                    .pitch_type\
                                    .value_counts(normalize=True)

            pitcher_dict[pitcher] = pitch_type_weights.to_dict()

        # Fill nan values
        pitcher_dict = pd.DataFrame(pitcher_dict).fillna(0).to_dict()


        # Select replacement pitch type and fill NaN values

        def pick_a_pitch(pitcher_id):
            """
            Returns a random pitch type label
            Uses pitchers prior pitch type probabilites as weights
            """

            population = list(pitcher_dict[pitcher_id].keys())
            weights = list(pitcher_dict[pitcher_id].values())

            return random.choices(population, weights, k=1)[0]

        # Iterate by instance, fill null values
        self.df['pitch_type'] = self.df.apply(
            lambda row: pick_a_pitch(row['pitcher']) \
                if pd.isnull(row['pitch_type']) \
                else row['pitch_type'], axis = 1)


    def pitch_subtype(self):
        """
        Creates a pitch_subtype feature
        """

        pitch_type_map = {'FA':'fastball', 'FF':'fastball', 'FT':'fastball', 'FC':'fastball',
                          'FS':'fastball', 'SI':'fastball', 'SF':'fastball', 'SL':'breaking',
                          'CB':'breaking', 'CU':'breaking', 'SC':'breaking', 'KC':'breaking',
                          'CH':'offspeed', 'KN':'offspeed', 'EP':'offspeed', 'FO':'breaking',
                          'PO':'pitchout', 'IN':'pitchout'}

        self.df['pitch_subtype'] = self.df['pitch_type']
        self.df['pitch_type'] = self.df['pitch_type'].map(pitch_type_map)


    def count_status(self):
        """
        Feature: count_status
        Description: The ratio of balls and strikes for the current at bat
        Issue: There are two existing features related to the count. We need to represent the count as a categorical feature.
        Solution: Classifiy the pitchers position reguarding the count (Ahead, Behind, Neutral)
        """

        self.df['balls'] = self.df['balls'].replace({4:3, 5:3})

        self.df['count_status'] = self.df['balls'].astype('int').astype('str')\
                                  + self.df['strikes'].astype('int').astype('str')

        count_status_mapping = {
            '00':'neutral', '21':'neutral', '32':'neutral', '10':'behind',
            '20':'behind', '30':'behind', '31':'behind', '01':'ahead',
            '02':'ahead', '11':'ahead', '12':'ahead', '22':'ahead'
        }

        self.df['count_status'] = self.df['count_status'].map(count_status_mapping)


    def score_differential(self):
        """
        Feature: Score Differential
        Description: The absolute value of the difference in home team score and away team score
        """

        self.df['score_differential'] = abs(self.df['home_score'] - self.df['away_score'])


    def bases_loaded(self):
        """
        Feature**: Bases Loaded
        Description: A binary indication of the bases being loaded or not
        """
        self.df['on_1b'] = self.df['on_1b'] * 0 + 1
        self.df['on_1b'] = self.df['on_1b'].fillna(0)
        self.df['on_2b'] = self.df['on_2b'] * 0 + 1
        self.df['on_2b'] = self.df['on_2b'].fillna(0)
        self.df['on_3b'] = self.df['on_3b'] * 0 + 1
        self.df['on_3b'] = self.df['on_3b'].fillna(0)

        self.df['bases_loaded'] = self.df['on_1b'] + self.df['on_2b'] + self.df['on_3b']
        self.df['bases_loaded'] = self.df['bases_loaded'].apply(lambda x: 1 if x == 3 else 0)


    def batter_swung(self):
        """
        Feature: swung
        Description: Binary feature describing wheather or not the batter swung at the pitch or not
        """

        swung = ['foul','hit_into_play','swinging_strike','hit_into_play_no_out',
                 'hit_into_play_score','foul_tip','swinging_strike_blocked',
                 'foul_bunt','missed_bunt']

        self.df['batter_swung'] = self.df['description'].apply(lambda x: 1 if x in swung else 0)


    def ball_position(self):
        """
        Creates a feature describing where the pitch crosses the strikezone plane
        """

        self.df['ball_high'] = self.df['plate_z'] > self.df['sz_top']
        self.df['ball_low'] = self.df['plate_z'] < self.df['sz_bot']
        self.df['ball_left'] = self.df['plate_x'].apply(lambda x: x < -0.73)
        self.df['ball_right'] = self.df['plate_x'].apply(lambda x: x > 0.73)


    def in_strikezone(self):
        """
        Binary feature representing wheather or not the pitch was in the strikezone
        """

        self.df['in_strikezone'] = (self.df['ball_high'].astype(int)
                                    + self.df['ball_low'].astype(int)
                                    + self.df['ball_left'].astype(int)
                                    + self.df['ball_right'].astype(int))

        self.df['in_strikezone'] = self.df['in_strikezone'].apply(
                                       lambda x: 0
                                           if x > 0
                                           else 1)


    def chased(self):
        """
        Binary feature representing wheather or not the batter chased the pitch
        """

        self.df['chased'] = self.df['batter_swung'] - self.df['in_strikezone']
        self.df['chased'] = self.df['chased'].apply(lambda x: 1 if x == 1 else 0)


    def pitcher_team(self):
        self.df['player_name'] = self.df['player_name'].str.replace(' ', '_')
        self.df['player_team'] = self.df['player_name'].apply(lambda x: Player(x).team)


    def clean(self):
        print('Dropping features...')
        self.drop_features()
        print('Done')
        print('Dropping instances...')
        self.drop_instances()
        print('Done')
        print('Typecasting...')
        self.fielding_alignment_typecast()
        print('Done')
        print('Sorting pitches...')
        self.chronological_sort()
        print('Done')
        print('Cleaning pitch type...')
        self.pitch_type()
        print('Done')
        print('Creating pitch subtype...')
        self.pitch_subtype()
        print('Done')
        print('Creating count status...')
        self.count_status()
        print('Done')
        print('Creating score differential...')
        self.score_differential()
        print('Done')
        print('Creating bases loaded...')
        self.bases_loaded()
        print('Done')
        print('Creating batter swung...')
        self.batter_swung()
        print('Done')
        print('Creating ball position...')
        self.ball_position()
        print('Done')
        print('Creating strikezone...')
        self.in_strikezone()
        print('Done')
        print('Creating chased...')
        self.chased()
        print('Done')
        print('Looking up player teams...')
        self.pitcher_team()
        print('Done')
        print('Data cleaning...DONE!')
        return self.df
