from database.repository import *
import mysql.connector
import re


class MysqlRepository(Repository):

    def __init__(self):
        super().__init__()
        config = {
            'user': 'root',
            'password': 'root',
            'host': 'localhost', # to run LOCALLY, this should be localhost to run with flask then db
            'port': '32000', # to run LOCALLY, this should be 32000 to run with flask then 3306
            'database': 'ikatadomain',
            'autocommit': True,
        }
        self.connection = mysql.connector.connect(**config)
        self.cursor = self.connection.cursor()

    def __del__(self):
        #self.cursor.close() I think this was a casualty of switching to mysql 8.0
        self.connection.close()

    def abstract(self):
        pass