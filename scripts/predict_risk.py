import pandas as pd
import joblib
import logging
import os
import glob
import requests  # Import requests for making API calls

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskPredictor:
    def __init__(self):
        self.model = self.load_latest_model()
        self.data_path = 'data/pipeline_data.csv'
        self.qwen_api_url = os.getenv("QWEN_API_URL")  # Set your Qwen API URL in environment variables
        self.qwen_api_key = os.getenv("QWEN_API_KEY")  # Set your Qwen API key in environment variables

    def load_latest_model(self):
        """Load the latest trained model from the models directory."""
        try:
            model_files = glob.glob('models/pipeline_model_*.pkl')
            if not model_files:
                raise FileNotFoundError("No trained models found")
                
            latest_model = sorted(model_files)[-1]
            logger.info(f"Loading model: {latest_model}")
            return joblib.load(latest_model)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def get_ai_suggestion(self, failure_type):
        """Get AI-powered remediation suggestion from Qwen AI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": f"Suggest specific technical solutions for: {failure_type}",
                "max_tokens": 150
            }
            # Added timeout parameter to fix B113
            response = requests.post(self.qwen_api_url, headers=headers, json=payload, timeout=30)  # Added timeout
            response.raise_for_status()  # Raise an error for bad responses
            return response.json().get("choices")[0].get("text").strip()
        except Exception as e:
            logger.error(f"Qwen AI suggestion failed: {str(e)}")
            return "AI suggestion unavailable"

    def predict_risks(self):
        try:
            data = pd.read_csv(self.data_path)
            required_features = ['build_time', 'test_count', 'hour_of_day',
                                'day_of_week', 'test_per_second']
            
            if not set(required_features).issubset(data.columns):
                missing = list(set(required_features) - set(data.columns))
                raise ValueError(f"Missing features: {missing}")
                
            predictions = self.model.predict_proba(data[required_features])
            results = []
            
            for idx, pred in enumerate(predictions):
                risk = {
                    "timestamp": data.iloc[idx]['timestamp'],
                    "failure_probability": pred[1],
                    "predicted_class": int(pred[1] > 0.5),
                    "ai_suggestion": self.get_ai_suggestion("build_failure" if pred[1] > 0.5 else "test_failure")
                }
                results.append(risk)
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

def main():
    try:
        predictor = RiskPredictor()
        results = predictor.predict_risks()
        
        # Save results
        results.to_csv('data/risk_predictions.csv', index=False)
        logger.info("Risk predictions saved with AI suggestions")
        
        # Print sample suggestion
        sample = results.iloc[0]
        logger.info(f"\nSample Prediction:\n"
                   f"Failure Probability: {sample['failure_probability']:.2%}\n"
                   f"AI Suggestion: {sample['ai_suggestion']}")
                   
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
