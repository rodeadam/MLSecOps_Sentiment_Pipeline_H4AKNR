"""
Unit tests for Mobile Sentiment Analysis Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from MLModel import SentimentMLModel
from constants import *
import os
import tempfile


@pytest.fixture
def sample_data():
    """Create sample training data"""
    data = {
        'review_id': range(1, 101),
        'customer_name': ['Customer ' + str(i) for i in range(1, 101)],
        'age': np.random.randint(18, 60, 100),
        'brand': np.random.choice(['Apple', 'Samsung', 'Google'], 100),
        'model': np.random.choice(['iPhone 14', 'Galaxy S24', 'Pixel 8'], 100),
        'price_usd': np.random.uniform(300, 1500, 100),
        'price_local': np.random.uniform(300, 1500, 100),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], 100),
        'exchange_rate_to_usd': np.random.uniform(0.8, 1.5, 100),
        'rating': np.random.randint(1, 6, 100),
        'review_text': ['Great phone!' for _ in range(100)],
        'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], 100),
        'country': np.random.choice(['USA', 'UK', 'Germany'], 100),
        'language': np.random.choice(['English', 'German'], 100),
        'review_date': pd.date_range('2023-01-01', periods=100).astype(str),
        'verified_purchase': np.random.choice([True, False], 100),
        'battery_life_rating': np.random.randint(1, 6, 100),
        'camera_rating': np.random.randint(1, 6, 100),
        'performance_rating': np.random.randint(1, 6, 100),
        'design_rating': np.random.randint(1, 6, 100),
        'display_rating': np.random.randint(1, 6, 100),
        'review_length': np.random.randint(20, 200, 100),
        'word_count': np.random.randint(5, 50, 100),
        'helpful_votes': np.random.randint(0, 20, 100),
        'source': np.random.choice(['Amazon', 'eBay', 'Flipkart'], 100)
    }
    return pd.DataFrame(data)


def test_model_initialization():
    """Test model initialization"""
    model = SentimentMLModel()
    assert model.model is None
    assert isinstance(model.label_encoders, dict)
    assert model.feature_names is None


def test_preprocessing(sample_data):
    """Test data preprocessing"""
    model = SentimentMLModel()
    X, y = model.preprocess_data(sample_data, is_training=True)
    
    # Check shapes
    assert X.shape[0] == 100
    assert y.shape[0] == 100
    
    # Check that ID and text columns are dropped
    assert ID_COLUMN not in X.columns
    assert TEXT_FEATURES[0] not in X.columns
    
    # Check that all features are float
    assert X.dtypes.apply(lambda x: x == 'float64').all()
    
    # Check target encoding (should be 0, 1, 2 for three classes)
    assert set(y) <= {0, 1, 2}


def test_model_training(sample_data):
    """Test model training"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow_uri = f"file://{tmpdir}/mlruns"
        
        model = SentimentMLModel()
        results = model.train(sample_data, mlflow_tracking_uri=mlflow_uri)
        
        # Check results
        assert 'run_id' in results
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert 'classification_report' in results
        
        # Check accuracy is reasonable
        assert 0.0 <= results['train_accuracy'] <= 1.0
        assert 0.0 <= results['test_accuracy'] <= 1.0
        
        # Check model is trained
        assert model.model is not None
        assert len(model.label_encoders) > 0


def test_model_prediction(sample_data):
    """Test model prediction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow_uri = f"file://{tmpdir}/mlruns"
        
        # Train model
        model = SentimentMLModel()
        model.train(sample_data, mlflow_tracking_uri=mlflow_uri)
        
        # Create inference data (first 10 rows without target)
        inference_data = sample_data.head(10).drop(columns=[TARGET_COLUMN])
        
        # Make predictions
        predictions = model.predict(inference_data)
        
        # Check predictions
        assert len(predictions) == 10
        assert set(predictions) <= set(SENTIMENT_CLASSES)


def test_invalid_data():
    """Test handling of invalid data"""
    model = SentimentMLModel()
    
    # Missing required columns
    invalid_data = pd.DataFrame({'random_col': [1, 2, 3]})
    
    with pytest.raises(Exception):
        model.preprocess_data(invalid_data, is_training=True)


def test_artifact_saving_and_loading(sample_data):
    """Test saving and loading artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        mlflow_uri = f"file://{tmpdir}/mlruns"
        
        # Train model
        model = SentimentMLModel()
        model.train(sample_data, mlflow_tracking_uri=mlflow_uri)
        
        # Check artifacts are saved
        assert os.path.exists('label_encoders.pkl')
        assert os.path.exists('scaler.pkl')
        assert os.path.exists('sentiment_encoder.pkl')
        
        # Create new model and load artifacts
        new_model = SentimentMLModel()
        new_model.load_artifacts(tmpdir)
        
        # Check artifacts are loaded
        assert len(new_model.label_encoders) > 0
        assert new_model.scaler is not None
        assert new_model.sentiment_encoder is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
