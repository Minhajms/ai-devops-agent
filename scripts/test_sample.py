import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def sample_data():
    # Adding timestamp column to match the model's expectations
    return pd.DataFrame({
        'timestamp': [datetime.now().timestamp() for _ in range(3)],
        'build_time': [120, 80, 150],
        'test_count': [50, 30, 60],
        'failure': [0, 1, 0]
    })

def test_model_training(sample_data):
    from train_model import PipelinePredictor
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = sample_data
        predictor = PipelinePredictor()
        predictor.train_model()
        
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')

def test_slack_notification():
    from notify_slack import SlackNotifier
    
    # Mock environment variable
    with patch.dict('os.environ', {'SLACK_TOKEN': 'dummy_token'}):
        with patch('slack_sdk.WebClient.chat_postMessage') as mock_post:
            mock_post.return_value = {'ok': True}
            notifier = SlackNotifier()
            response = notifier.send_message("Test message")
            assert response['ok'] is True
