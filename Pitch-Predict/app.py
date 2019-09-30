import os
import random
import pickle
import datetime

from .games import today, query_redshift, Player
from .Bauer_model import Machine_learning
from .column_names import col_names
from .preprocessing import Preprocess
from .ML import query_redshift, Boost_tree

import pandas as pd
import numpy as np
import psycopg2
from flask import Flask, request, render_template
import category_encoders
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


def create_app():
    """Create and configure instance of flask application"""
    app = Flask(__name__, instance_relative_config=True)

    """
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        """


    @app.route('/')
    def home():
        nav_links = ['Todays Games', 'Seasons', 'Teams', 'Players', 'About']
        return render_template('base.html', title = 'Home', birds = nav_links)


    @app.route('/about')
    def about():
        return "Devloped by Matt Kirby, Josh Mancuso, and Nicklaus Winters"


    @app.route('/today')
    def todays_stuff():
        today_df = today()
        todays_games = today_df['todays_games'].tolist()
        todays_home_pitcher_ids = today_df['home_pitchers'].tolist()
        todays_home_pitcher_name = today_df['home_pitchers_names'].tolist()
        todays_away_pitcher_ids = today_df['away_pitchers'].tolist()
        todays_home_pitcher_name = today_df['away_pitchers_names'].tolist()

        return str(todays_games[0]) + str(today_df['home_pitchers'][0])


    @app.route('/predict')
    def predict():
        # Import models
        f = open('Pitch-Predict/Bauer_ordinal_multiclass_best_models_v1.pkl',
         'rb')

        models_df = pd.read_pickle(f,compression='zip')

        x_train, x_test, y_train, y_test = Machine_learning().tts()

        models_df = models_df.sort_values(by = 'accuracy', ascending = False)
        model = models_df['model'][0]

        model.fit(x_train, y_train)

        sample_pitch = x_test.sample(1)
        prediction = model.predict(sample_pitch)

        pitch_type_map = {'FA':1, 'FF':1, 'FT':2, 'FC':2, 'FS':2, 'SI':2, 'SF':2, 'N/A':2.5, 'SL':3,
                          'CB':4, 'CU':4, 'SC':5, 'KC':5, 'CH':6, 'KN':7, 'EP':8, 'FO':9, 'PO':9}

        prediction_map = dict((v,k) for k,v in pitch_type_map.items())

        return render_template('predict.html',
                               sample_pitch = sample_pitch,
                               prediction = prediction_map[prediction[0]])


    @app.route('/train/<id>')
    def train(id):

        sample_id = 425844

        pitcher_df = Boost_tree(id)
        x_train, x_test, y_train, y_test = Boost_tree.split(pitcher_df.df)
        model = Boost_tree.model(x_train, y_train)

        return str(model.predict(x_test.sample(1)))


    return app
