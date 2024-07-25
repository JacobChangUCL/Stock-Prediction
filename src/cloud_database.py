"""
this module is for uploading data to AWS RDS MySQL database and
querying data from AWS RDS
"""
import mysql.connector
import pandas as pd


# Connect to database
def connect_to_database():
    """
    Connect to AWS RDS MySQL database
    """
    try:
        mydb = mysql.connector.connect(
            host="database-1.c100seoimp1h.ap-southeast-2.rds.amazonaws.com",
            # the endport of my RDS
            user="admin",  # RDS database username
            password="12345678",  # RDS database password
            database="Daps"  # the name of the database
        )
    except Exception as e:
        print("Error: ", e, "\nThe AWS RDS MySQL database is expired,need to pay for it.")
        return None
    return mydb


def insert_data(database, table: str, data):
    """
        Insert data into table
        these method could cause SQL injection risk,
        If you want to use this method, 
        you need to make sure that the data isn't too valuable
    """
    inserted_data = ""
    for index, row in data.iterrows():
        inserted_data = inserted_data + "(" + ", ".join(
            [str(i) if isinstance(i, (int, float)) else f"'{i}'" for i in row]) + "),"
    inserted_data = inserted_data[:-1]
    # transfer the data into string, if the data is not int or float type, add '' to it to satisfy the SQL syntax
    cursor1 = database.cursor()
    query = f"INSERT INTO {table} VALUES {inserted_data}"
    cursor1.execute(query)
    database.commit()
    print("inserted data finished")
    return True


def download_data(table_name, database: mysql.connector.MySQLConnection):
    """
    download data from AWS RDS MySQL database
    """
    cursor = database.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()
    return data


# create table
SQL_CREAT_AAPL = """
CREATE TABLE AAPL (
    Date DATE  PRIMARY KEY,
    Open DOUBLE,
    High DOUBLE,
    Low DOUBLE,
    Close DOUBLE,
    Adj_Close DOUBLE,
    Volume DOUBLE
)DEFAULT CHARSET=utf8"""


def cloud_database():
    mydb = connect_to_database()
    if mydb is None:
        return False

    cursor = mydb.cursor()
    cursor.execute("SHOW TABLES LIKE 'AAPL'")
    result = cursor.fetchone()
    if result:
        print("Table exists.")
    else:
        print("Table does not exist, creating table.")
        cursor.execute(SQL_CREAT_AAPL)  # create table
        AAPL_data = pd.read_csv("./data/AAPL.csv", parse_dates=['Date'])
        AAPL_data['Date'] = AAPL_data['Date'].dt.date
        insert_data(mydb, 'AAPL', AAPL_data)

    AAPL = download_data("AAPL", mydb)
    print("data from cloud database:\n", AAPL[:10], '\n')
    mydb.close()
    return True
