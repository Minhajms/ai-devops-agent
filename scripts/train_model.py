import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelinePredictor:
    def __init__(self, data_path="data/pipeline_data.csv"):
        self.data_path = data_path
        self.model = None
        self.model_path = "models"
        self.create_model_directory()
        self.smote = os.getenv('SMOTE_SAMPLING', 'false').lower() == 'true'

    def create_model_directory(self):
        os.makedirs(self.model_path, exist_ok=True)

    def load_and_preprocess_data(self):
        try:
            logger.info(f"Loading data from {self.data_path}")
            data = pd.read_csv(self.data_path)
            
            # Feature engineering
            data['hour_of_day'] = pd.to_datetime(data['timestamp'], unit='s').dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp'], unit='s').dt.dayofweek
            data['test_per_second'] = data['test_count'] / data['build_time']
            data['failure'] = data['failure'].astype(int)
            
            return data
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            raise

    def handle_class_imbalance(self, X, y):
        if self.smote and sum(y) > 1:
            logger.info("Applying SMOTE for class imbalance")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        return X, y

    def train_model(self):
        try:
            data = self.load_and_preprocess_data()
            features = ['build_time', 'test_count', 'hour_of_day', 
                       'day_of_week', 'test_per_second']
            X = data[features]
            y = data['failure']

            # Handle class imbalance
            X_balanced, y_balanced = self.handle_class_imbalance(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42
            )

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'class_weight': ['balanced', None]
            }
            
            model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1_weighted'
            )
            model.fit(X_train, y_train)
            
            self.model = model.best_estimator_
            logger.info(f"Best parameters: {model.best_params_}")
            
            # Evaluation
            y_pred = self.model.predict(X_test)
            logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            
            # Save model and metadata
            self._save_model_artifacts(model.best_params_)
            
            return model.best_score_
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _save_model_artifacts(self, params):
        """Save model and metadata"""
        model_name = "pipeline_model_latest.pkl"
        meta_name = "model_metadata.json"
        
        # Save model
        joblib.dump(self.model, os.path.join(self.model_path, model_name))
        
        # Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "features_used": ['build_time', 'test_count', 'hour_of_day', 
                             'day_of_week', 'test_per_second'],
            "best_params": params,
            "model_type": "RandomForestClassifier"
        }
        
        with open(os.path.join(self.model_path, meta_name), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model and metadata saved to {self.model_path}")

if __name__ == "__main__":
    try:
        predictor = PipelinePredictor()
        score = predictor.train_model()
        logger.info(f"Model training complete with best score: {score:.4f}")
    except Exception as e:
        logger.error(f"Critical error in training: {str(e)}")
        exit(1)
