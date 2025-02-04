import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import os
from datetime import datetime

# Enhanced logging
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
        
    def create_model_directory(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
    def load_and_preprocess_data(self):
        try:
            logger.info(f"Loading data from {self.data_path}")
            data = pd.read_csv(self.data_path)
            
            # Handle missing timestamp column if needed
            if 'timestamp' not in data.columns:
                logger.warning("Timestamp column not found, adding current timestamp")
                data['timestamp'] = datetime.now().timestamp()
            
            # Add derived features
            data['hour_of_day'] = pd.to_datetime(data['timestamp'], unit='s').dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp'], unit='s').dt.dayofweek
            data['test_per_second'] = data['test_count'] / data['build_time']
            
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def train_model(self):
        try:
            data = self.load_and_preprocess_data()
            
            # Feature selection
            features = ['build_time', 'test_count', 'hour_of_day', 
                       'day_of_week', 'test_per_second']
            X = data[features]
            y = data['failure']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            logger.info("Training model...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            y_pred = self.model.predict(X_test)
            
            # Log performance metrics
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Average CV score: {cv_scores.mean():.3f}")
            logger.info("\nClassification Report:\n" + 
                       classification_report(y_test, y_pred))
            
            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"pipeline_model_{timestamp}.pkl"
            model_path = os.path.join(self.model_path, model_filename)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save feature importance analysis
            self._save_feature_importance(features)
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def _save_feature_importance(self, features):
        importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        logger.info("\nFeature Importance:\n" + str(importance))
        importance.to_csv(os.path.join(self.model_path, 'feature_importance.csv'))
