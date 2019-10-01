import pandas as pd
from pybaseball import statcast
import random
from mr_clean import Season_cleaner, Player


seasons = {'2010':{'start_date': '2010-04-04', 'end_date': '2010-11-01'},
           '2011':{'start_date': '2011-03-31', 'end_date': '2011-10-28'},
           '2012':{'start_date': '2012-03-28', 'end_date': '2012-10-28'},
           '2013':{'start_date': '2013-03-31', 'end_date': '2013-10-30'},
           '2014':{'start_date': '2014-03-22', 'end_date': '2014-10-29'},
           '2015':{'start_date': '2015-04-05', 'end_date': '2015-11-01'},
           '2016':{'start_date': '2016-04-03', 'end_date': '2016-11-02'},
           '2017':{'start_date': '2017-04-02', 'end_date': '2017-11-01'},
           '2018':{'start_date': '2018-03-29', 'end_date': '2018-10-28'},
           '2019':{'start_date': '2019-03-20', 'end_date': '2019-09-07'}
           }


def pull_statcast_data(start_date, end_date, year):
    """
    Date Format: YYYY-MM-DD
    """
    df = statcast(start_dt = start_date, end_dt = end_date)
    return df


def compress_and_export(df, year, f_path = "Season_CSVs/"):
    """
    Pickle DataFrame
    """
    df.to_csv(path=(f_path + year + ".csv"))


def main():

    def pull_clean_and_export(start_date, end_date, year):
        """
        Queries statcast, calls cleaning function, pickles season dataframes,
        and writes to seasons directory
        """
        df = pull_statcast_data(start_date, end_date, year)
        df = Season_cleaner(df).clean()
        compress_and_export(df, year)

    for year in seasons.keys():
        start_date = seasons[year]['start_date']
        end_date = seasons[year]['end_date']
        pull_clean_and_export(start_date, end_date, year)


if __name__ == "__main__":
    main()
