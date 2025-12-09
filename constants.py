"""
Constants for Mobile Reviews Sentiment Analysis ML Pipeline
"""

# Target column
TARGET_COLUMN = 'sentiment'

# ID column to drop
ID_COLUMN = 'review_id'

# Categorical features
CATEGORICAL_FEATURES = [
    'brand',
    'model',
    'currency',
    'country',
    'language',
    'source',
    'customer_name'
]

# Numerical features
NUMERICAL_FEATURES = [
    'age',
    'price_usd',
    'price_local',
    'exchange_rate_to_usd',
    'rating',
    'battery_life_rating',
    'camera_rating',
    'performance_rating',
    'design_rating',
    'display_rating',
    'review_length',
    'word_count',
    'helpful_votes'
]

# Binary features
BINARY_FEATURES = [
    'verified_purchase'
]

# Date features (to extract temporal features from)
DATE_FEATURES = [
    'review_date'
]

# Text features (for NLP processing)
TEXT_FEATURES = [
    'review_text'
]

# Feature groups for preprocessing
FEATURES_TO_ENCODE = CATEGORICAL_FEATURES
FEATURES_TO_SCALE = NUMERICAL_FEATURES

# All features used in model (excluding ID, target, and raw text)
ALL_FEATURES = (
    CATEGORICAL_FEATURES + 
    NUMERICAL_FEATURES + 
    BINARY_FEATURES
)

# Sentiment classes
SENTIMENT_CLASSES = ['Negative', 'Neutral', 'Positive']

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'objective': 'multi:softmax',
    'num_class': 3,
    'random_state': 42
}

# Test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42
