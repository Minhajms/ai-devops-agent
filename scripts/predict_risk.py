import pandas as pd
import joblib
import logging
import os
import glob
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls: int, period: int = 60):
        self.calls = calls
        self.period = period
        self.timestamps = []

    def wait(self):
        now = time.time()
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.timestamps[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.timestamps = self.timestamps[1:]
        
        self.timestamps.append(now)

class RiskPredictor:
    def __init__(self):
        self.model = self.load_latest_model()
        self.data_path = 'data/pipeline_data.csv'
        self.qwen_api_url = os.getenv("QWEN_API_URL")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.rate_limiter = RateLimiter(calls=10, period=60)  # 10 calls per minute
        self.suggestion_cache = {}

    def load_latest_model(self) -> Any:
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

    def get_cached_suggestion(self, failure_type: str) -> Optional[str]:
        cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
        return self.suggestion_cache.get(cache_key)

    def get_ai_suggestion(self, failure_type: str) -> str:
        cached = self.get_cached_suggestion(failure_type)
        if cached:
            return cached

        try:
            self.rate_limiter.wait()
            
            headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }
            
            context = self._get_failure_context(failure_type)
            
            payload = {
                "prompt": f"Given this context: {context}\nSuggest specific technical solutions for: {failure_type}",
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.qwen_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            suggestion = response.json().get("choices")[0].get("text").strip()
            
            # Cache the suggestion
            cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
            self.suggestion_cache[cache_key] = suggestion
            
            return suggestion
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._get_fallback_suggestion(failure_type)
        except Exception as e:
            logger.error(f"Unexpected error in AI suggestion: {str(e)}")
            return self._get_fallback_suggestion(failure_type)

    def _get_failure_context(self, failure_type: str) -> str:
        try:
            df = pd.read_csv(self.data_path)
            recent_failures = df[df['failure'] == 1].tail(5)
            return f"Recent failure patterns: {recent_failures.to_dict(orient='records')}"
        except Exception:
            return "No historical context available"

    def _get_fallback_suggestion(self, failure_type: str) -> str:
        fallback_suggestions = {
            "build_failure": "Check for dependency conflicts and ensure all required packages are listed in requirements.txt",
            "test_failure": "Review recent test failures and check for environmental differences between local and CI",
            "default": "Review logs and recent changes for potential issues"
        }
        return fallback_suggestions.get(failure_type, fallback_suggestions["default"])

    def predict_risks(self) -> pd.DataFrame:
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
                failure_prob = pred[1]
                failure_type = self._determine_failure_type(data.iloc[idx], failure_prob)
                
                risk = {
                    "timestamp": data.iloc[idx]['timestamp'],
                    "failure_probability": failure_prob,
                    "predicted_class": int(failure_prob > 0.5),
                    "failure_type": failure_type,
                    "ai_suggestion": self.get_ai_suggestion(failure_type)
                }
                results.append(risk)
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _determine_failure_type(self, row: pd.Series, failure_prob: float) -> str:
        if failure_prob <= 0.5:
            return "low_risk"
        if row['test_count'] > row['test_count'].mean():
            return "test_failure"
        return "build_failure"

def main():
    try:
        predictor = RiskPredictor()
        results = predictor.predict_risks()
        
        # Save results
        os.makedirs('data', exist_ok=True)
        results.to_csv('data/risk_predictions.csv', index=False)
        
        # Print summary
        high_risk = results[results['failure_probability'] > 0.7]
        if not high_risk.empty:
            logger.warning(f"Found {len(high_risk)} high-risk predictions!")
            for _, risk in high_risk.iterrows():
                logger.warning(
                    f"High risk detected:\n"
                    f"Probability: {risk['failure_probability']:.2%}\n"
                    f"Type: {risk['failure_type']}\n"
                    f"Suggestion: {risk['ai_suggestion']}"
                )
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
