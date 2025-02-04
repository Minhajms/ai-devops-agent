import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
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
    # Assuming the model expects certain features
    features = data[['build_time', 'test_count', 'hour_of_day', 'day_of_week', 'test_per_second']]
    predictions = model.predict(features)
    return predictions

def main():
    # Load the model
    model_path = 'models/pipeline_model_latest.pkl'  # Adjust the path as necessary
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Load the data
    data_path = 'data/pipeline_data.csv'  # Adjust the path as necessary
    data = load_data(data_path)

    # Predict risk
    risk_predictions = predict_risk(model, data)
    logger.info(f"Risk predictions: {risk_predictions}")

    # Optionally, save predictions to a file
    predictions_df = pd.DataFrame(risk_predictions, columns=['Risk Prediction'])
    predictions_df.to_csv('data/risk_predictions.csv', index=False)
    logger.info("Risk predictions saved to data/risk_predictions.csv")

if __name__ == "__main__":
    main()
