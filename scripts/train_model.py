import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
        """Ensure the models directory exists."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def load_and_preprocess_data(self):
        """Load and preprocess pipeline data."""
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
        """Train a predictive model using the pipeline data."""
        try:
            data = self.load_and_preprocess_data()
            
            # Feature selection
            features = ['build_time', 'test_count', 'hour_of_day', 'day_of_week', 'test_per_second']
            X = data[features]
            y = data['failure']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
            
            # Evaluate model
            score = self.model.score(X_test, y_test)
            logger.info(f"Test set score: {score:.3f}")
            
            # Generate classification report
            y_pred = self.model.predict(X_test)
            logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            
            # Save model with timestamp
            self._save_model()
            
            # Save feature importance analysis
            self._save_feature_importance(features)
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _save_model(self):
        """Save the trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"pipeline_model_{timestamp}.pkl"
        model_path = os.path.join(self.model_path, model_filename)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def _save_feature_importance(self, features):
        """Save feature importance analysis."""
        importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        logger.info("\nFeature Importance:\n" + str(importance))
        importance.to_csv(os.path.join(self.model_path, 'feature_importance.csv'), index=False)

if __name__ == "__main__":
    predictor = PipelinePredictor()
    predictor.train_model()
