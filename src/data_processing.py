import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, data_path, processed_data):
        self.data_path = data_path
        self.processed_data = processed_data
        # Ensure the processed data directory exists
        os.makedirs(self.processed_data, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=mutual_info_classif, k=10)

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
        Process the data by handling data types, missing values, feature engineering, and feature selection.
        """
        try:
            logger.info("Starting data processing...")

            # Label encode Efficiency_Status
            self.df['Efficiency_Status'] = self.label_encoder.fit_transform(self.df['Efficiency_Status'])
            logger.info("Efficiency_Status label encoded: %s", 
                       dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))))

            # Drop unnecessary columns
            self.df = self.df.drop(['Timestamp', 'Machine_ID'], axis=1)

            # One-hot encode Operation_Mode
            self.df = pd.get_dummies(self.df, columns=['Operation_Mode'], prefix='Mode')

            # Feature Engineering: Interaction terms
            self.df['Predictive_Maintenance_Mode_Active'] = self.df['Predictive_Maintenance_Score'] * self.df['Mode_Active']
            self.df['Quality_Control_Mode_Active'] = self.df['Quality_Control_Defect_Rate_Perc'] * self.df['Mode_Active']
            self.df['Network_Latency_Mode_Active'] = self.df['Network_Latency_ms'] * self.df['Mode_Active']

            # Bin Predictive_Maintenance_Score
            self.df['Predictive_Maintenance_Binned'] = pd.cut(
                self.df['Predictive_Maintenance_Score'], bins=3, labels=['Low', 'Medium', 'High']
            )
            self.df = pd.get_dummies(self.df, columns=['Predictive_Maintenance_Binned'], prefix='PM_Bin')

            # Split features and target
            X = self.df.drop('Efficiency_Status', axis=1)
            y = self.df['Efficiency_Status']

            # Split data: 10% holdout, 18% test, 18% validation, 54% train
            X_temp, X_holdout, y_temp, y_holdout = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
            logger.info("Data split: 54%% train, 18%% validation, 18%% test, 10%% holdout")

            # Store indices for later use
            X_train_index = X_train.index
            X_val_index = X_val.index
            X_test_index = X_test.index
            X_holdout_index = X_holdout.index

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train_index)

            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val_index)

            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test_index)

            X_holdout_scaled = self.scaler.transform(X_holdout)
            X_holdout_scaled = pd.DataFrame(X_holdout_scaled, columns=X_holdout.columns, index=X_holdout_index)

            # Feature Selection
            X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
            selected_features = X_train_scaled.columns[self.selector.get_support()].tolist()
            logger.info("Selected Features: %s", selected_features)

            X_val_selected = self.selector.transform(X_val_scaled)
            X_test_selected = self.selector.transform(X_test_scaled)
            X_holdout_selected = self.selector.transform(X_holdout_scaled)

            # Apply SMOTE to training data only
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
            logger.info("Balanced Training Class Distribution:\n%s", 
                       pd.Series(y_train_balanced).value_counts().to_dict())

            # Compute sample weights for balanced training data
            class_counts = pd.Series(y).value_counts().sort_index()  # High=0, Low=1, Medium=2
            weights = {0: 10, 1: 1, 2: class_counts[1]/class_counts[2]}
            sample_weights = np.array([weights[label] for label in y_train_balanced])

            # Save processed data and preprocessors
            joblib.dump(X_train_balanced, os.path.join(self.processed_data, 'X_train_balanced.pkl'))
            joblib.dump(y_train_balanced, os.path.join(self.processed_data, 'y_train_balanced.pkl'))
            joblib.dump(sample_weights, os.path.join(self.processed_data, 'sample_weights.pkl'))
            joblib.dump(X_val_selected, os.path.join(self.processed_data, 'X_val_selected.pkl'))
            joblib.dump(y_val, os.path.join(self.processed_data, 'y_val.pkl'))
            joblib.dump(X_test_selected, os.path.join(self.processed_data, 'X_test_selected.pkl'))
            joblib.dump(y_test, os.path.join(self.processed_data, 'y_test.pkl'))
            joblib.dump(X_holdout_selected, os.path.join(self.processed_data, 'X_holdout_selected.pkl'))
            joblib.dump(y_holdout, os.path.join(self.processed_data, 'y_holdout.pkl'))
            joblib.dump(self.scaler, os.path.join(self.processed_data, 'scaler.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.processed_data, 'label_encoder.pkl'))
            joblib.dump(self.selector, os.path.join(self.processed_data, 'selector.pkl'))
            joblib.dump(selected_features, os.path.join(self.processed_data, 'selected_features.pkl'))

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