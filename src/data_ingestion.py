import os
import sys
from dotenv import load_dotenv
import mysql.connector
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, RAW_FILE_PATH
# Ensure the raw directory exists
if not os.path.exists(RAW_DIR):
    os.makedirs(RAW_DIR)


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        try:
            logger.info("Initializing DataIngestion")
            load_dotenv()
            self.mysql_host = os.getenv("MYSQL_HOST", "localhost")
            self.mysql_user = os.getenv("MYSQL_USER", "root")
            self.mysql_password = os.getenv("MYSQL_PASSWORD", "")
            self.mysql_database = os.getenv("MYSQL_DATABASE", "smart_manufacturing")
            logger.info(f"Loaded MySQL configuration: host={self.mysql_host}, user={self.mysql_user}, database={self.mysql_database}")
        except Exception as e:
            raise CustomException("Error initializing DataIngestion", sys.exc_info()) from e
        
    def connect_mysql(self):
        try:
            logger.info("Connecting to MySQL database")
            connection = mysql.connector.connect(
                    host=self.mysql_host,
                    user=self.mysql_user,
                    password=self.mysql_password,
                    database=self.mysql_database
                )
            logger.info("Successfully connected to MySQL")
            return connection
        except mysql.connector.Error as e:
            raise CustomException(f"Error connecting to MySQL: {e}", sys.exc_info())
            


        # fetch the data from mysql database and save as csv file in data directory    
    def fetch_data_to_csv(self, query, output_file):
        try:
            connection = self.connect_mysql()
            cursor = connection.cursor()
            logger.info(f"Executing query: {query}")
            cursor.execute(query)
            data = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(output_file, index=False)
            logger.info(f"Data fetched and saved to {output_file}")
        except Exception as e:
                raise CustomException("Error fetching data to CSV", sys.exc_info()) from e
        finally:
                cursor.close()
                connection.close()

if __name__ == "__main__":
    
    data_ingestion = DataIngestion()

    data_ingestion.connect_mysql()

    data_ingestion.fetch_data_to_csv(query="SELECT * FROM machine_efficiency", output_file=RAW_FILE_PATH)
    logger.info("Data ingestion completed successfully.")


