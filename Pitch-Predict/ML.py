from .column_names import col_names
from .preprocessing import Preprocess

import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

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


class Boost_tree:


    def __init__(self, pitcher_id):

        # Query Redshift DB
        self.query = "SELECT * FROM pitches\
                      WHERE pitcher = " + str(pitcher_id)

        # Create pitcher df
        self.df = pd.DataFrame(query_redshift(self.query),
                               columns = col_names)


    def split(df):

        # Preprocess df
        processor = Preprocess(dataframe = df)
        df = processor.process(processor.df)

        # Encode pitch_type
        target_mapping = {
            'fastball': 0 ,
            'breaking': 1,
            'offspeed': 2
        }

        # Create feature matrix and target vector
        feature_matrix = df.drop(columns = ['next_pitch'])
        target_vector = df['next_pitch'].map(target_mapping).fillna(0)

        # Train test split
        x_train, x_test, y_train, y_test = train_test_split(feature_matrix,
                                                            target_vector)

        return x_train, x_test, y_train, y_test


    def model(x_train, y_train):

        # Instantiate classifier
        clf = Pipeline(steps = [('scaler', RobustScaler()),
                                ('boost', GradientBoostingClassifier())])

        # Grid search param grid
        params = {
            'boost__max_depth': [4]
        }

        # Instantiate hyperparameter tuning
        grid = GridSearchCV(
            estimator = clf,
            scoring = 'balanced_accuracy',
            param_grid = params,
            refit = True,
            cv = 2,
            verbose = 10,
            n_jobs = -1)

        # Train grid
        grid.fit(x_train, y_train)

        # Select best model
        model = grid.best_estimator_

        return model
