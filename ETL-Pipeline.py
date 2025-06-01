import os
import sys
import platform
from dotenv import load_dotenv
import mysql.connector
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException

# Initialize logger
logger = get_logger(__name__)

# Load environment variables for MySQL
load_dotenv()
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "smart_manufacturing")

logger.info(f"Loaded MySQL configuration: host={MYSQL_HOST}, user={MYSQL_USER}, database={MYSQL_DATABASE}")

class ManufacturingData:
    def __init__(self, data):
        try:
            logger.info("Initializing ManufacturingData")
            self.data = data
        except Exception as e:
            raise CustomException("Error initializing ManufacturingData", sys.exc_info()) from e

    def process_data(self, file_path):
        try:
            logger.info(f"Reading CSV file: {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV with {len(data)} rows")

            # Transformation: Convert Timestamp to datetime with error handling
            logger.info("Converting Timestamp column to datetime with errors='coerce'")
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
            logger.debug("Timestamp conversion completed")

            # Log any rows with invalid timestamps (converted to NaT)
            invalid_timestamps = data['Timestamp'].isna().sum()
            if invalid_timestamps > 0:
                logger.warning(f"Found {invalid_timestamps} invalid timestamps, converted to NaT")

            data.reset_index(drop=True, inplace=True)
            logger.debug("Reset index of DataFrame")

            return data
        except Exception as e:
            raise CustomException("Error processing CSV data", sys.exc_info()) from e

    def create_mysql_table(self, cursor, table):
        try:
            logger.info(f"Creating table {table} in MySQL if it doesn't exist")
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                Timestamp DATETIME,
                Machine_ID INT,
                Operation_Mode VARCHAR(50),
                Temperature_C FLOAT,
                Vibration_Hz FLOAT,
                Power_Consumption_kW FLOAT,
                Network_Latency_ms FLOAT,
                Packet_Loss_Perc FLOAT,
                Quality_Control_Defect_Rate_Perc FLOAT,
                Production_Speed_units_per_hr FLOAT,
                Predictive_Maintenance_Score FLOAT,
                Error_Rate_Perc FLOAT,
                Efficiency_Status VARCHAR(50)
            )
            """
            cursor.execute(create_table_query)
            logger.info(f"Table {table} created or already exists")
        except Exception as e:
            raise CustomException("Error creating MySQL table", sys.exc_info()) from e

    def ensure_database_exists(self):
        try:
            # Connect to MySQL without specifying a database
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD
            )
            cursor = conn.cursor()

            # Check if the database exists
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            if self.database not in databases:
                logger.info(f"Database {self.database} does not exist, creating it")
                cursor.execute(f"CREATE DATABASE {self.database}")
                logger.info(f"Database {self.database} created successfully")
            else:
                logger.info(f"Database {self.database} already exists")

            cursor.close()
            conn.close()
        except Exception as e:
            raise CustomException("Error ensuring database exists", sys.exc_info()) from e

    def insert_data_mysql(self, data, database, table):
        self.database = database
        self.table = table
        self.data = data

        # Ensure the database exists before connecting
        self.ensure_database_exists()

        try:
            logger.info(f"Connecting to MySQL database: {database}, table: {table}")
            # Connect to the specific database
            self.mysql_conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=self.database
            )
            self.cursor = self.mysql_conn.cursor()
            logger.info("Connected to MySQL database")

            # Create table if it doesn't exist
            self.create_mysql_table(self.cursor, self.table)

            # Prepare the INSERT query
            insert_query = f"""
            INSERT INTO {self.table} (
                Timestamp, Machine_ID, Operation_Mode, Temperature_C, Vibration_Hz,
                Power_Consumption_kW, Network_Latency_ms, Packet_Loss_Perc,
                Quality_Control_Defect_Rate_Perc, Production_Speed_units_per_hr,
                Predictive_Maintenance_Score, Error_Rate_Perc, Efficiency_Status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Insert records into MySQL directly from DataFrame
            logger.info(f"Inserting {len(data)} records into MySQL")
            for _, row in data.iterrows():
                # Convert NaT to None for MySQL
                timestamp = row['Timestamp'] if pd.notna(row['Timestamp']) else None
                values = (
                    timestamp,
                    row['Machine_ID'],
                    row['Operation_Mode'],
                    row['Temperature_C'],
                    row['Vibration_Hz'],
                    row['Power_Consumption_kW'],
                    row['Network_Latency_ms'],
                    row['Packet_Loss_%'],
                    row['Quality_Control_Defect_Rate_%'],
                    row['Production_Speed_units_per_hr'],
                    row['Predictive_Maintenance_Score'],
                    row['Error_Rate_%'],
                    row['Efficiency_Status']
                )
                self.cursor.execute(insert_query, values)

            # Commit the transaction
            self.mysql_conn.commit()
            logger.info(f"Successfully inserted {len(data)} records into MySQL")

            return len(self.data)
        except Exception as e:
            if 'mysql_conn' in dir(self):
                self.mysql_conn.rollback()
            raise CustomException("Error inserting data into MySQL", sys.exc_info()) from e
        finally:
            if 'cursor' in dir(self):
                self.cursor.close()
            if 'mysql_conn' in dir(self):
                self.mysql_conn.close()
            logger.info("MySQL connection closed")

if __name__ == "__main__":
    try:
        logger.info("Starting ETL-Pipeline.py script")
        file_path = "notebook\\data.csv"
        database = "smart_manufacturing"
        table = "machine_efficiency"
        logger.info(f"Configuration: file_path={file_path}, database={database}, table={table}")

        logger.info("Creating ManufacturingData instance")
        manufacturing_data = ManufacturingData(file_path)

        logger.info("Processing CSV data")
        processed_data = manufacturing_data.process_data(file_path)

        logger.info(f"Processed {len(processed_data)} records")

        logger.info("Inserting records into MySQL")
        inserted_count = manufacturing_data.insert_data_mysql(processed_data, database, table)

        logger.info(f"Successfully inserted {inserted_count} records into MySQL")
        print(f"Inserted {inserted_count} records into MySQL.")

    except Exception as e:
        raise CustomException("Script failed", sys.exc_info()) from e