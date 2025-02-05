import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def sample_data():
    # Generate more test data to support cross-validation
    n_samples = 20  # Increased number of samples
    return pd.DataFrame({
        'timestamp': [datetime.now().timestamp() for _ in range(n_samples)],
        'build_time': np.random.randint(60, 180, n_samples),
        'test_count': np.random.randint(20, 80, n_samples),
        'failure': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 30% failure rate
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'test_per_second': np.random.uniform(0.3, 0.8, n_samples)
    })

def test_model_training(sample_data):
    from train_model import PipelinePredictor

    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = sample_data
        predictor = PipelinePredictor()
        predictor.train_model()

        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')

        # Test prediction functionality
        X_test = sample_data[['build_time', 'test_count', 'hour_of_day',
                            'day_of_week', 'test_per_second']].iloc[:1]
        prediction = predictor.model.predict(X_test)
        assert prediction in [0, 1]

def test_predict_risk(sample_data):
    from predict_risk import predict_risk
    from train_model import PipelinePredictor

    predictor = PipelinePredictor()
    predictor.train_model()
    model = predictor.model

    risk_predictions = predict_risk(model, sample_data)
    assert len(risk_predictions) == len(sample_data)
