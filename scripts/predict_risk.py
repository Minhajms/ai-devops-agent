import pandas as pd
import joblib
import logging
import os

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load pipeline data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def predict_risk(model, data):
    """Predict the risk based on the model and input data."""
    required_features = ['build_time', 'test_count', 'hour_of_day', 'day_of_week', 'test_per_second']
    missing_features = [feat for feat in required_features if feat not in data.columns]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        raise ValueError(f"Input data is missing required features: {missing_features}")

    features = data[required_features]
    predictions = model.predict(features)
    return predictions

def main():
    # Load the model
    model_path = 'models/pipeline_model_latest.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Attempting to train a new model...")
        from train_model import PipelinePredictor
        predictor = PipelinePredictor()
        predictor.train_model()
        model = predictor.model
    else:
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

    # Load the data
    data_path = 'data/pipeline_data.csv'
    data = load_data(data_path)

    # Predict risk
    risk_predictions = predict_risk(model, data)
    logger.info(f"Risk predictions: {risk_predictions}")

    # Save predictions to a file
    predictions_df = pd.DataFrame(risk_predictions, columns=['Risk Prediction'])
    predictions_df.to_csv('data/risk_predictions.csv', index=False)
    logger.info("Risk predictions saved to data/risk_predictions.csv")

if __name__ == "__main__":
    main()
