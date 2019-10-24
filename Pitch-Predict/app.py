import os
import random
import pickle
import datetime

from .games import *
from .column_names import col_names
from .preprocessing import Preprocess
from .ML import query_redshift, Boost_tree

import pandas as pd
import numpy as np
import psycopg2
from flask import Flask, request, render_template, jsonify
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


    @app.route('/')
    def home():
        nav_links = ['Todays Games', 'Seasons', 'Teams', 'Players', 'About']
        return render_template('base.html', title = 'Home', birds = nav_links)


    @app.route('/about')
    def about():
        return "Devloped by Matt Kirby, Brad Mortenson, Josh Mancuso, and Nicklaus Winters"


    @app.route('/pitchers/<input_date>')
    def pitchers_on_date(input_date):
        day = datetime.datetime(year=int(input_date[0:4]),
                                month=int(input_date[4:6]),
                                day=int(input_date[6:8]))
        pitchers = get_all_pitchers(day)
        return jsonify(pitchers) if len(pitchers) > 0 else 'No Games On That Date'


    @app.route('/today-pitchers')
    def todays_pitchers():
        today = datetime.date.today()
        todays_pitchers = get_all_pitchers(today)
        return jsonify(todays_pitchers) if len(todays_pitchers) > 0 else 'No Games Today'


    @app.route('/today-teams')
    def todays_games():
        today = datetime.date.today()
        todays_games = get_games(today)

        return jsonify(get_teams(todays_games)) if len(todays_games) > 0 else 'No Games Today'


    @app.route('/train/<id>')
    def train(id):
        pitcher_df = Boost_tree(id)
        x_train, x_test, y_train, y_test = Boost_tree.split(pitcher_df.df)
        model = Boost_tree.model(x_train, y_train)
        return str(model.predict(x_test.sample(1)))


    return app
