import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import PROCESSED_DIR, MODEL_DIR

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path, model_dir):
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir  # Directory where models are saved
        self.best_model = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.sample_weights = None
        self.X_val_selected = None
        self.y_val = None
        self.X_test_selected = None
        self.y_test = None
        self.X_holdout_selected = None
        self.y_holdout = None
        self.label_encoder = None
        self.selected_features = None

        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model training initialized with processed data at {self.processed_data_path} and model save path at {self.model_dir}")

    def load_data(self):
        """
        Load processed data from the specified path.
        """
        try:
            logger.info(f"Loading processed data from {self.processed_data_path}")
            self.X_train_balanced = joblib.load(os.path.join(self.processed_data_path, 'X_train_balanced.pkl'))
            self.y_train_balanced = joblib.load(os.path.join(self.processed_data_path, 'y_train_balanced.pkl'))
            self.sample_weights = joblib.load(os.path.join(self.processed_data_path, 'sample_weights.pkl'))
            self.X_val_selected = joblib.load(os.path.join(self.processed_data_path, 'X_val_selected.pkl'))
            self.y_val = joblib.load(os.path.join(self.processed_data_path, 'y_val.pkl'))
            self.X_test_selected = joblib.load(os.path.join(self.processed_data_path, 'X_test_selected.pkl'))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, 'y_test.pkl'))
            self.X_holdout_selected = joblib.load(os.path.join(self.processed_data_path, 'X_holdout_selected.pkl'))
            self.y_holdout = joblib.load(os.path.join(self.processed_data_path, 'y_holdout.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.processed_data_path, 'label_encoder.pkl'))
            self.selected_features = joblib.load(os.path.join(self.processed_data_path, 'selected_features.pkl'))
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Processed data files not found: {e}")
            raise CustomException("Processed data files not found", sys.exc_info()) from e

    def tune_hyperparameters(self):
        """
        Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.
        """
        try:
            logger.info("Starting hyperparameter tuning for XGBoost...")
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 1.0]
            }
            model_xgb = XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                random_state=42,
                eval_metric='mlogloss'
            )
            random_search = RandomizedSearchCV(
                estimator=model_xgb,
                param_distributions=param_dist,
                n_iter=10,
                cv=5,
                scoring='f1_macro',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(self.X_train_balanced, self.y_train_balanced, sample_weight=self.sample_weights)
            
            self.best_model = random_search.best_estimator_
            logger.info(f"Best parameters for XGBoost: {random_search.best_params_}")
            logger.info(f"Best cross-validation macro F1-score for XGBoost: {random_search.best_score_:.4f}")
            
            return random_search.best_score_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning for XGBoost: {e}")
            raise CustomException("Error during hyperparameter tuning for XGBoost", sys.exc_info()) from e

    def evaluate_model(self, X, y, set_name):
        """
        Evaluate the model on the specified dataset.
        """
        try:
            logger.info(f"Evaluating XGBoost on {set_name} set...")
            y_pred = self.best_model.predict(X)
            class_names = ['High', 'Low', 'Medium']  # Based on notebook's label encoding

            report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
            cm = confusion_matrix(y, y_pred)

            # Extract weighted averages and per-class F1-scores
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            class_f1_scores = {class_name: report[class_name]['f1-score'] for class_name in class_names}

            # Log results
            logger.info(f"XGBoost evaluation results on {set_name} set:\n"
                        f"Weighted Precision: {weighted_precision:.4f}\n"
                        f"Weighted Recall: {weighted_recall:.4f}\n"
                        f"Weighted F1 Score: {weighted_f1:.4f}\n"
                        f"Per-Class F1 Scores: {class_f1_scores}\n"
                        f"Confusion Matrix:\n{cm}")

            return {
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'per_class_f1': class_f1_scores,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
        except Exception as e:
            logger.error(f"Error during evaluation on {set_name} set: {e}")
            raise CustomException(f"Error during evaluation on {set_name} set", sys.exc_info()) from e

    def train_and_evaluate(self):
        """
        Train the XGBoost model and evaluate on validation, test, and holdout sets.
        """
        try:
            logger.info("Starting XGBoost model training and evaluation...")
            self.load_data()

            # Use best parameters from logs to avoid redundant tuning
            best_params = {
                'subsample': 0.8,
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.3
            }
            self.best_model = XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                random_state=42,
                eval_metric='mlogloss',
                **best_params
            )
            logger.info(f"Using pre-tuned parameters for XGBoost: {best_params}")

            # Train the model
            self.best_model.fit(self.X_train_balanced, self.y_train_balanced, sample_weight=self.sample_weights)
            logger.info("XGBoost model trained successfully on full training data")

            # Save the best model
            model_save_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            joblib.dump(self.best_model, model_save_path)
            logger.info(f"Best model saved at {model_save_path}")

            # Evaluate on validation, test, and holdout sets
            evaluation_results = {}
            evaluation_results['validation'] = self.evaluate_model(self.X_val_selected, self.y_val, "Validation")
            evaluation_results['test'] = self.evaluate_model(self.X_test_selected, self.y_test, "Test")
            evaluation_results['holdout'] = self.evaluate_model(self.X_holdout_selected, self.y_holdout, "Holdout")

            # Log feature importance
            feature_importance = pd.Series(self.best_model.feature_importances_, index=self.selected_features).sort_values(ascending=False)
            logger.info(f"Feature Importance (XGBoost):\n{feature_importance.to_dict()}")

            logger.info("Model training and evaluation completed successfully.")
            return evaluation_results

        except Exception as e:
            logger.error(f"Error during model training and evaluation: {e}")
            raise CustomException("Error during model training and evaluation", sys.exc_info()) from e

    def run(self):
        """
        Run the model training and evaluation process.
        """
        try:
            logger.info("Starting model training and evaluation process...")
            evaluation_results = self.train_and_evaluate()
            logger.info(f"Model training and evaluation results: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            logger.error(f"Error in model training run: {e}")
            raise CustomException("Error in model training run", sys.exc_info()) from e

if __name__ == "__main__":
    model_trainer = ModelTraining(
        processed_data_path=PROCESSED_DIR,
        model_dir=MODEL_DIR
    )
    results = model_trainer.run()
    logger.info(f"Model training and evaluation results: {results}")