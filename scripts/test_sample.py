import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the scripts directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_risk import RiskPredictor
from scripts.train_model import PipelinePredictor

@pytest.fixture
def sample_data():
    n_samples = 20
    return pd.DataFrame({
        'timestamp': [datetime.now().timestamp() for _ in range(n_samples)],
        'build_time': np.random.randint(60, 180, n_samples),
        'test_count': np.random.randint(20, 80, n_samples),
        'failure': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'test_per_second': np.random.uniform(0.3, 0.8, n_samples)
    })

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3] for _ in range(20)])
    return model

def test_risk_predictor_initialization(tmp_path):
    # Create a temporary model file
    model_path = tmp_path / "models"
    model_path.mkdir()
    mock_model_file = model_path / "pipeline_model_latest.pkl"
    mock_model_file.write_bytes(b"mock model data")
    
    with patch('joblib.load') as mock_load:
        mock_load.return_value = MagicMock()
        predictor = RiskPredictor()
        assert predictor.model is not None

def test_predict_risks(sample_data, mock_model):
    with patch('joblib.load', return_value=mock_model):
        predictor = RiskPredictor()
        with patch.object(predictor, 'get_ai_suggestion', return_value='Mock suggestion'):
            results = predictor.predict_risks()
            assert isinstance(results, pd.DataFrame)
            assert 'failure_probability' in results.columns
            assert 'ai_suggestion' in results.columns

def test_get_ai_suggestion():
    predictor = RiskPredictor()
    # Test with API failure (should return fallback suggestion)
    suggestion = predictor.get_ai_suggestion('build_failure')
    assert isinstance(suggestion, str)
    assert len(suggestion) > 0

def test_model_training(sample_data):
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = sample_data
        predictor = PipelinePredictor()
        
        # Mock SMOTE to avoid sklearn.utils import error
        with patch('imblearn.over_sampling.SMOTE') as mock_smote:
            mock_smote.return_value.fit_resample.return_value = (
                sample_data.drop('failure', axis=1),
                sample_data['failure']
            )
            predictor.train_model()

        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')

def test_determine_failure_type():
    predictor = RiskPredictor()
    row = pd.Series({
        'test_count': 100,
        'build_time': 120,
        'hour_of_day': 14,
        'day_of_week': 2,
        'test_per_second': 0.8
    })
    
    # Test low risk case
    assert predictor._determine_failure_type(row, 0.3) == 'low_risk'
    
    # Test high risk cases
    assert predictor._determine_failure_type(row, 0.8) in ['test_failure', 'build_failure']
