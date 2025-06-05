import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, data_path, processed_data):
        self.data_path = data_path
        self.processed_data = processed_data

        # Ensure the processed data directory exists
        os.makedirs(self.processed_data, exist_ok=True)  # Fixed: Create the directory itself

    def load_data(self):
        """
        Load data from the specified path.
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
            return self.df
        except Exception as e:
            raise CustomException("Error loading data", sys.exc_info()) from e

    def process_data(self, data):
        """
        Process the data by handling data types, missing values, and outliers.
        """
        try:
            logger.info("Starting data processing...")
            # Convert timestamp to datetime
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')

            # Convert categorical columns to 'category' dtype
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col] = self.df[col].astype('category')

            # Extract year, month, day, hour from 'Timestamp'
            self.df['Year'] = self.df['Timestamp'].dt.year
            self.df['Month'] = self.df['Timestamp'].dt.month
            self.df['Day'] = self.df['Timestamp'].dt.day
            self.df['Hour'] = self.df['Timestamp'].dt.hour

            # Convert categorical columns to numerical using Label Encoding
            label_encoders = {}
            categorical_cols = self.df.select_dtypes(include=['category']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le

            # Drop unnecessary columns and columns with no impact on the model
            self.df.drop(columns=['Timestamp', 'Machine_ID', 'Error_Rate_Perc', 'Production_Speed_units_per_hr', 'Year'], inplace=True)  # Fixed: Correct column name

            X = self.df.drop(columns=['Efficiency_Status'])
            y = self.df['Efficiency_Status']

            # Scale X features with StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split the data into training and testing sets and save as joblib files
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            logger.info("Data split into training and testing sets.")

            joblib.dump(X_train, os.path.join(self.processed_data, 'X_train.pkl'))
            joblib.dump(X_test, os.path.join(self.processed_data, 'X_test.pkl'))
            joblib.dump(y_train, os.path.join(self.processed_data, 'y_train.pkl'))
            joblib.dump(y_test, os.path.join(self.processed_data, 'y_test.pkl'))

            joblib.dump(scaler, os.path.join(self.processed_data, 'scaler.pkl'))

            logger.info("Data processing completed successfully.")

        except Exception as e:
            raise CustomException("Error processing data", sys.exc_info()) from e

    def run(self):
        """
        Run the data processing pipeline.
        """
        try:
            data = self.load_data()
            self.process_data(data)
            logger.info("Data processing pipeline completed successfully.")
        except Exception as e:
            raise CustomException("Error in data processing pipeline", sys.exc_info()) from e

if __name__ == "__main__":
    data_processor = DataProcessing(data_path=RAW_FILE_PATH, processed_data=PROCESSED_DIR)
    data_processor.run()
    logger.info("Data processing script executed successfully.")