import sqlite3
import pandas as pd

class SQLiteDB:
    """
    Builds and/or connects to SQLite database
    """

    def __init__(self, path):
        """
        Summary line.

        Constructor for SQLiteDB class

        Parameters
        ----------
        path : str
            location of SQLiteDB

        """
        self.__path = path
        self.__connection = self.__create_connection()
        self.__available_import_file_types = {
            "excel": 2,
            "csv": 1,
            "txt": 0
        }
        
    def __create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.__path)
            return conn
        except sqlite3.Error as e:
            print(e)
        return conn

    def execute_select_stmt(self, query):
        conn = self.__connection
        rows = pd.read_sql_query(query, conn)
        return rows
    
    def execute_select_create_stmt(self, query, tableName, drop_if_exist=True):
        dataframe = self.execute_select_stmt(query)
        ## If tableName already exists... then drop
        if drop_if_exist:
            self.drop_table(tableName)

        dataframe.to_sql(name=tableName, con=self.__connection, index=False)
        print(f"Successfully created {tableName}")

    def drop_table(self, table_name):
        conn = self.__connection
        cur = conn.cursor()
        cur.execute(f'DROP TABLE IF EXISTS {table_name}')
        print(f'Successfully dropped table {table_name}')
        cur.close()

    def import_file(self, file_path, table_name, file_type="txt"):
        print('Reading file...')

        file_code = self.__available_import_file_types.get(file_type, lambda: "Invalid file type")

        if file_code == 0:
            dataframe = pd.read_table(file_path, encoding="ISO-8859-1")
        elif file_code == 1:
            dataframe = pd.read_csv(file_path, encoding="ISO-8859-1")
        elif file_code == 2:
            dataframe = pd.read_excel(file_path)
		
        self.drop_table(table_name)
        dataframe.to_sql(name=table_name, con=self.__connection, index=False)
        # Maybe do a count for how many records were inserted
        print(f'Successfully imported {table_name}')

    #specifically pandas df
    def import_table(self, table_data, table_name):
        print('Reading_table')
        dataframe = table_data
        dataframe.to_sql(name=table_name, con=self.__connection)
        print('Successfully imported file into table')


    def __convert_sql_term(self, string_term):
        """
        purpose:
            Insert variables to f string
        input:
            string_term - a string value
        returns:
            f string
        """
        return f"`{string_term}`"

    def execute_query(self, columns_to_use, table_to_query, target_feature, where_stmt):
        """
        purpose:
            Assembles SELECT SQL query and executes it against specified SQL table
        input:
            columns_to_use: List of strings for column names for both categorical and numerical columns
            table_to_query: String value of table_name to be queried
            target_feature (optional): String value of target feature (for supervised learning) to be added as column to be queried
            where_stmt (optional): String value of SQL where conditions
        returns:
            pandas dataframe of returned rows
        """
        print('Connecting to db...')
        columns_to_use_as_string = ', '.join(map(self.__convert_sql_term, columns_to_use)) # Maps columns to readable format for sql

        if target_feature != 0:
            target_feature_string = ', ' + target_feature
        else:
            target_feature_string = ''

        if where_stmt != 0:
            where_stmt_string = where_stmt
        else:
            where_stmt_string = ''

        prepared_stmt = f'SELECT {columns_to_use_as_string} {target_feature_string} FROM {table_to_query} {where_stmt_string}'

        dataset = self.execute_select_stmt(prepared_stmt)

        dataset[columns_to_use] = dataset[columns_to_use].apply(pd.to_numeric, errors='coerce')
        print('Successfully obtained data from db')
        return dataset
