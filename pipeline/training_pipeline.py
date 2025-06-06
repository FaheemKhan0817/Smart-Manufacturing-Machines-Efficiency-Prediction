import sys
import os
from datetime import datetime
from src.data_processing import DataProcessing
from src.data_ingestion import DataIngestion
from src.model_training import ModelTraining
from src.custom_exception import CustomException
from config.paths_config import RAW_FILE_PATH, PROCESSED_DIR, MODEL_DIR
from src.logger import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline execution at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Step 1: Data Ingestion
        logger.info("Starting data ingestion...")
        data_ingestion = DataIngestion()
        data_ingestion.connect_mysql()
        data_ingestion.fetch_data_to_csv(query="SELECT * FROM machine_efficiency", output_file=RAW_FILE_PATH)
        logger.info("Data ingestion completed successfully.")

        # Step 2: Data Processing
        logger.info("Starting data processing...")
        data_processor = DataProcessing(data_path=RAW_FILE_PATH, processed_data=PROCESSED_DIR)
        data_processor.run()
        logger.info("Data processing script executed successfully.")

        # Step 3: Model Training
        logger.info("Starting model training...")
        model_trainer = ModelTraining(
            processed_data_path=PROCESSED_DIR,
            model_dir=MODEL_DIR
        )
        results = model_trainer.run()
        logger.info(f"Model training and evaluation results: {results}")

        logger.info("Training pipeline execution completed successfully at %s", 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except CustomException as ce:
        logger.error(f"Custom exception occurred in training pipeline: {ce}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred in training pipeline: {e}")
        raise CustomException("Error in training pipeline execution", sys.exc_info()) from e