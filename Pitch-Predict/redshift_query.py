import psycopg2
import numpy as np

con = psycopg2.connect(dbname= 'dev', host='examplecluster.cdbpwaymevt5.us-east-2.redshift.amazonaws.com',
                       port= '5439', user= 'awsuser', password= '******')

cur = con.cursor()

cur.execute("""SELECT * FROM pitches
               WHERE pitcher = 425844""")

print(np.array(cur.fetchall()))

cur.close() 
con.close()

"""query = SELECT COUNT * FROM pitches
           WHERE pitcher = 425844


def query_redshift(query_string):
    con = psycopg2.connect(
        dbname= 'dev',
        host='examplecluster.cdbpwaymevt5.us-east-2.redshift.amazonaws.com',
        port= '5439',
        user= 'awsuser',
        password= '******')

    cur = con.cursor()

    cur.execute(query_string)

    results = cur.fetchall()

    cur.close()
    con.close()

    return results

print(np.array(query_redshift(query)))"""
