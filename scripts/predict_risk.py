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
        self.model = None
        self.data_path = 'data/pipeline_data.csv'
        self.qwen_api_url = os.getenv("QWEN_API_URL")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.rate_limiter = RateLimiter(calls=10, period=60)
        self.suggestion_cache = {}
        self.historical_thresholds = self._calculate_historical_thresholds()
        self._initialize_model()

    def _initialize_model(self):
        """Safe model initialization with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.model = self.load_latest_model()
                return
            except Exception as e:
                logger.error(f"Model initialization failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to initialize model after multiple attempts")

    def _calculate_historical_thresholds(self) -> Dict[str, float]:
        """Calculate historical averages with fallback"""
        try:
            if os.path.exists(self.data_path):
                df = pd.read_csv(self.data_path)
                return {
                    'avg_test_count': df['test_count'].median(),
                    'avg_build_time': df['build_time'].quantile(0.75)
                }
        except Exception as e:
            logger.warning(f"Historical threshold calculation failed: {str(e)}")
        
        return {
            'avg_test_count': 50,
            'avg_build_time': 120
        }

    def load_latest_model(self) -> Any:
        try:
            model_files = glob.glob('models/pipeline_model_*.pkl')
            if not model_files:
                raise FileNotFoundError("No trained models found")
                
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading model: {latest_model}")
            
            # Verify model file integrity
            if os.path.getsize(latest_model) == 0:
                raise ValueError("Model file is empty/corrupted")
                
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
            
            # Validate API response structure
            if not response.json().get("choices"):
                raise ValueError("Unexpected API response format")
                
            suggestion = response.json()["choices"][0].get("text", "").strip()
            if not suggestion:
                raise ValueError("Empty suggestion received")
            
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
            if not os.path.exists(self.data_path):
                return "No historical data available"
                
            df = pd.read_csv(self.data_path)
            if df.empty:
                return "Empty historical data"
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            recent_data = df[df['date'] > (datetime.now() - timedelta(days=30))]
            
            if recent_data.empty:
                return "No recent data available"
                
            return f"Last 30 days stats:\n- Avg tests: {recent_data['test_count'].mean():.1f}\n- Build success rate: {1 - recent_data['failure'].mean():.1%}"
        
        except Exception as e:
            logger.warning(f"Context gathering failed: {str(e)}")
            return "No historical context available"

    def _determine_failure_type(self, row: pd.Series, failure_prob: float) -> str:
        if failure_prob <= 0.5:
            return "low_risk"
            
        if pd.isna(row['test_count']) or pd.isna(row['build_time']):
            return "data_quality_issue"
            
        if row['test_count'] > self.historical_thresholds['avg_test_count'] * 1.5:
            return "test_failure"
        elif row['build_time'] > self.historical_thresholds['avg_build_time'] * 1.3:
            return "build_failure"
        return "environment_failure"

    def predict_risks(self) -> pd.DataFrame:
        try:
            # Validate data source
            if not os.path.exists(self.data_path):
                logger.error("Missing pipeline data file: data/pipeline_data.csv")
                return pd.DataFrame()
                
            data = pd.read_csv(self.data_path)
            
            # Validate data quality
            if data.empty:
                logger.warning("Empty pipeline data file")
                return pd.DataFrame()
                
            required_features = ['build_time', 'test_count', 'hour_of_day',
                                'day_of_week', 'test_per_second']
            
            # Check for required columns
            missing_features = set(required_features) - set(data.columns)
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return pd.DataFrame()
            
            # Clean numerical data
            data[required_features] = data[required_features].apply(
                pd.to_numeric, errors='coerce'
            )
            data = data.dropna(subset=required_features)
            
            if data.empty:
                logger.warning("No valid data after cleaning")
                return pd.DataFrame()
            
            # Make predictions
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
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _cache_suggestion(self, failure_type: str, suggestion: str):
        cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
        self.suggestion_cache[cache_key] = suggestion
        # Maintain cache size
        if len(self.suggestion_cache) > 100:
            self.suggestion_cache.pop(next(iter(self.suggestion_cache)), None)

    def _get_cached_suggestion(self, failure_type: str) -> Optional[str]:
        cache_key = f"{failure_type}_{datetime.now().strftime('%Y%m%d')}"
        return self.suggestion_cache.get(cache_key)

    def _get_fallback_suggestion(self, failure_type: str) -> str:
        fallback_suggestions = {
            "build_failure": (
                "1. Verify dependency versions in requirements.txt\n"
                "2. Check build system configuration\n"
                "3. Review memory/cpu usage during build"
            ),
            "test_failure": (
                "1. Investigate flaky tests\n"
                "2. Check test environment consistency\n"
                "3. Review test data setup/teardown"
            ),
            "environment_failure": (
                "1. Validate CI/CD environment variables\n"
                "2. Check external service availability\n"
                "3. Review infrastructure health metrics"
            ),
            "data_quality_issue": (
                "1. Check data collection process\n"
                "2. Validate pipeline metrics format\n"
                "3. Review data source integrations"
            ),
            "default": (
                "1. Check system logs for errors\n"
                "2. Review recent configuration changes\n"
                "3. Validate pipeline step execution order"
            )
        }
        return fallback_suggestions.get(failure_type, fallback_suggestions["default"])

def main():
    try:
        predictor = RiskPredictor()
        results = predictor.predict_risks()
        
        if not results.empty:
            os.makedirs('data', exist_ok=True)
            results.to_csv('data/risk_predictions.csv', index=False)
            
            high_risk = results[results['failure_probability'] > 0.7]
            if not high_risk.empty:
                logger.warning(f"High-risk predictions detected: {len(high_risk)}")
                for _, risk in high_risk.iterrows():
                    logger.warning(
                        f"High Risk Alert - Type: {risk['failure_type']}\n"
                        f"Probability: {risk['failure_probability']:.2%}\n"
                        f"Suggestion: {risk['ai_suggestion']}"
                    )
        else:
            logger.warning("No risk predictions generated")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
