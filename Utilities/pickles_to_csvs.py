## Takes Pickle Files and Exports Them to CSVs
import pandas as pd
source_path = "Season_pickles/"
destination_path = ""

filenames = [str(i) + '.pkl' for i in range(2010,2020)]

seasons = ['df_' + str(i) for i in range(10,20)]

season_dataframes = {}

for i in list(zip(filenames, seasons)):
        path = source_path + i[0]
        df = pd.read_pickle(path, compression='zip')
        df = df.drop(columns = ['des'])
        season_dataframes[i[1]] = df


i = 2010
for df in season_dataframes.values():
    path = destination_path + str(i) + ".csv"
    df.to_csv(path)
    print(path[-16:], '...Done')
    i += 1
