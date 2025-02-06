import pandas as pd
import joblib
import logging
import os
import glob
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

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
        self.rate_limiter = RateLimiter(calls=10, period=60)
        self.suggestion_cache = {}
        self.historical_thresholds = self._calculate_historical_thresholds()

    def _calculate_historical_thresholds(self) -> Dict[str, float]:
        """Calculate historical averages for key metrics"""
        try:
            df = pd.read_csv(self.data_path)
            return {
                'avg_test_count': df['test_count'].mean(),
                'avg_build_time': df['build_time'].mean()
            }
        except Exception:
            return {
                'avg_test_count': 50,  # Fallback values
                'avg_build_time': 120
            }

    def load_latest_model(self) -> Any:
        try:
            model_files = glob.glob('models/pipeline_model_*.pkl')
            if not model_files:
                raise FileNotFoundError("No trained models found")
                
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading model: {latest_model}")
            return joblib.load(latest_model)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def get_ai_suggestion(self, failure_type: str) -> str:
        cached = self._get_cached_suggestion(failure_type)
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
                "prompt": f"CI/CD failure context: {context}\nSuggest technical solutions for: {failure_type}",
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
            self._cache_suggestion(failure_type, suggestion)
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
            # Limit to last 30 days for relevant context
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            recent_data = df[df['date'] > (datetime.now() - timedelta(days=30))]
            recent_failures = recent_data[recent_data['failure'] == 1]
            return f"Last 30 days failure patterns: {recent_failures.describe().to_dict()}"
        except Exception as e:
            logger.warning(f"Context gathering failed: {str(e)}")
            return "No historical context available"

    def _determine_failure_type(self, row: pd.Series, failure_prob: float) -> str:
        if failure_prob <= 0.5:
            return "low_risk"
            
        if row['test_count'] > self.historical_thresholds['avg_test_count']:
            return "test_failure"
        elif row['build_time'] > self.historical_thresholds['avg_build_time']:
            return "build_failure"
        return "environment_failure"

    def predict_risks(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.data_path)
            required_features = ['build_time', 'test_count', 'hour_of_day',
                               'day_of_week', 'test_per_second']
            
            # Validate data quality
            if data.empty:
                raise ValueError("Empty dataset provided")
                
            missing_features = set(required_features) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Ensure numeric types
            data[required_features] = data[required_features].apply(pd.to_numeric, errors='coerce')
            data = data.dropna(subset=required_features)
            
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

    def _cache_suggestion(self, failure_type: str, suggestion: str):
        cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
        self.suggestion_cache[cache_key] = suggestion
        # Keep cache size under control
        if len(self.suggestion_cache) > 100:
            oldest_key = next(iter(self.suggestion_cache))
            del self.suggestion_cache[oldest_key]

    def _get_cached_suggestion(self, failure_type: str) -> Optional[str]:
        cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
        return self.suggestion_cache.get(cache_key)

    def _get_fallback_suggestion(self, failure_type: str) -> str:
        fallback_suggestions = {
            "build_failure": "1. Check dependency versions\n2. Verify build configuration\n3. Check system resources",
            "test_failure": "1. Review test environment setup\n2. Check test data validity\n3. Verify test isolation",
            "environment_failure": "1. Check CI/CD environment variables\n2. Verify service availability\n3. Review infrastructure health",
            "default": "1. Review recent changes\n2. Check system logs\n3. Validate configuration files"
        }
        return fallback_suggestions.get(failure_type, fallback_suggestions["default"])

def main():
    try:
        predictor = RiskPredictor()
        results = predictor.predict_risks()
        
        os.makedirs('data', exist_ok=True)
        results.to_csv('data/risk_predictions.csv', index=False)
        
        high_risk = results[results['failure_probability'] > 0.7]
        if not high_risk.empty:
            logger.warning(f"High-risk predictions detected: {len(high_risk)}")
            for _, risk in high_risk.iterrows():
                logger.warning(f"""
                HIGH RISK ALERT
                Timestamp: {risk['timestamp']}
                Probability: {risk['failure_probability']:.2%}
                Type: {risk['failure_type']}
                Suggestion: {risk['ai_suggestion']}
                """)
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
