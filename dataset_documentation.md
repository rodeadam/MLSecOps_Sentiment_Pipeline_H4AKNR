# Mobile Reviews Sentiment Dataset Documentation

## Overview

The Mobile Reviews Sentiment dataset contains 50,000 mobile phone reviews collected from various e-commerce platforms across multiple countries. Each review includes detailed product information, customer demographics, ratings, and sentiment labels.

## Dataset Statistics

- **Total Records**: 50,000 reviews
- **Time Period**: 2022-2025
- **Countries**: 10 (USA, UK, Germany, Brazil, India, Canada, Australia, UAE, etc.)
- **Languages**: 5 (English, German, Portuguese, Hindi, etc.)
- **Brands**: 7 major smartphone manufacturers
- **Sources**: 5 e-commerce platforms

## Target Variable

### sentiment
- **Type**: Categorical (Multi-class)
- **Classes**: 3
  - `Positive`: Favorable reviews (rating typically 4-5)
  - `Negative`: Unfavorable reviews (rating typically 1-2)
  - `Neutral`: Mixed or moderate reviews (rating typically 3)
- **Distribution**: 
  - Positive: ~60%
  - Negative: ~20%
  - Neutral: ~20%

## Features (25 columns)

### 1. Identifier

#### review_id
- **Type**: Integer
- **Description**: Unique identifier for each review
- **Range**: 1 to 50,000
- **Usage**: Dropped during preprocessing (not used for prediction)

### 2. Customer Demographics

#### customer_name
- **Type**: String
- **Description**: Customer identifier or name
- **Encoding**: Label encoded
- **Example**: "John Smith", "Maria Silva", "Amit Patel"
- **Usage**: Used as a categorical feature

#### age
- **Type**: Integer
- **Description**: Customer age in years
- **Range**: 18-60
- **Mean**: ~35
- **Usage**: Numerical feature (scaled)

#### country
- **Type**: Categorical
- **Description**: Customer's country
- **Values**: USA, UK, Germany, Brazil, India, Canada, Australia, UAE, France, etc.
- **Encoding**: Label encoded
- **Usage**: Important feature for regional preferences

#### language
- **Type**: Categorical
- **Description**: Review language
- **Values**: English, German, Portuguese, Hindi, French
- **Encoding**: Label encoded
- **Usage**: May correlate with sentiment expression patterns

### 3. Product Information

#### brand
- **Type**: Categorical
- **Description**: Smartphone brand
- **Values**: Apple, Samsung, Google, Xiaomi, OnePlus, Motorola, Realme
- **Encoding**: Label encoded
- **Usage**: Critical feature - brand perception affects sentiment

#### model
- **Type**: Categorical
- **Description**: Specific phone model
- **Examples**: 
  - Apple: iPhone 13, iPhone 14, iPhone 15 Pro, iPhone SE
  - Samsung: Galaxy S24, Galaxy A55, Galaxy Note 20
  - Google: Pixel 6, Pixel 7a, Pixel 8
  - Xiaomi: Mi 13 Pro, Redmi Note 13, Poco X6
  - OnePlus: OnePlus 12, OnePlus Nord 3
  - Motorola: Moto G Power, Edge 50, Razr 40
  - Realme: Realme 12 Pro, Realme Narzo 70
- **Encoding**: Label encoded
- **Usage**: Model-specific features influence sentiment

### 4. Pricing Information

#### price_usd
- **Type**: Float
- **Description**: Price in USD
- **Range**: $150 - $1500
- **Mean**: ~$650
- **Usage**: Numerical feature (scaled) - value perception

#### price_local
- **Type**: Float
- **Description**: Price in local currency
- **Range**: Varies by currency
- **Usage**: Numerical feature (scaled)

#### currency
- **Type**: Categorical
- **Description**: Local currency code
- **Values**: USD, EUR, GBP, INR, BRL, CAD, AUD, AED
- **Encoding**: Label encoded
- **Usage**: Regional economic context

#### exchange_rate_to_usd
- **Type**: Float
- **Description**: Exchange rate to USD
- **Range**: 0.78 - 83.0
- **Usage**: Numerical feature (scaled)

### 5. Rating Information

#### rating
- **Type**: Integer
- **Description**: Overall product rating
- **Range**: 1-5 stars
- **Distribution**:
  - 5 stars: ~35%
  - 4 stars: ~30%
  - 3 stars: ~20%
  - 2 stars: ~10%
  - 1 star: ~5%
- **Usage**: **Highly correlated with sentiment** - primary predictive feature

#### battery_life_rating
- **Type**: Integer
- **Description**: Battery performance rating
- **Range**: 1-5
- **Usage**: Numerical feature - component satisfaction

#### camera_rating
- **Type**: Integer
- **Description**: Camera quality rating
- **Range**: 1-5
- **Usage**: Numerical feature - critical for many users

#### performance_rating
- **Type**: Integer
- **Description**: System performance rating
- **Range**: 1-5
- **Usage**: Numerical feature - speed and responsiveness

#### design_rating
- **Type**: Integer
- **Description**: Physical design and aesthetics rating
- **Range**: 1-5
- **Usage**: Numerical feature - build quality perception

#### display_rating
- **Type**: Integer
- **Description**: Screen quality rating
- **Range**: 1-5
- **Usage**: Numerical feature - display satisfaction

### 6. Review Metadata

#### review_text
- **Type**: String (Text)
- **Description**: Full review text content
- **Length**: Variable (20-200 characters)
- **Examples**:
  - "Absolutely love this phone! The camera is next level."
  - "Battery drains too fast even on standby."
  - "Does what it's supposed to, nothing special."
- **Usage**: **Currently dropped** - potential for future NLP features (TF-IDF, embeddings)

#### review_date
- **Type**: Date (String)
- **Description**: Date when review was posted
- **Format**: YYYY-MM-DD
- **Range**: 2022-10-22 to 2025-10-13
- **Usage**: Extracted into temporal features:
  - `review_year`: 2022-2025
  - `review_month`: 1-12
  - `review_day`: 1-31
  - `review_dayofweek`: 0-6 (Monday=0)

#### review_length
- **Type**: Integer
- **Description**: Total character count of review text
- **Range**: 20-200
- **Mean**: ~65
- **Usage**: Numerical feature - review detail level

#### word_count
- **Type**: Integer
- **Description**: Number of words in review
- **Range**: 5-50
- **Mean**: ~12
- **Usage**: Numerical feature - verbosity indicator

#### helpful_votes
- **Type**: Integer
- **Description**: Number of users who found review helpful
- **Range**: 0-20
- **Mean**: ~5
- **Usage**: Numerical feature - review quality/credibility

#### verified_purchase
- **Type**: Boolean
- **Description**: Whether purchase was verified by platform
- **Values**: True/False
- **Distribution**: ~70% True
- **Usage**: Binary feature (converted to 0/1) - review authenticity

#### source
- **Type**: Categorical
- **Description**: E-commerce platform where review was posted
- **Values**: Amazon, eBay, Flipkart, BestBuy, AliExpress
- **Encoding**: Label encoded
- **Usage**: Platform-specific rating biases

## Feature Engineering

### Derived Features

From `review_date`, we extract:
1. **review_year**: Captures temporal trends
2. **review_month**: Seasonal patterns
3. **review_day**: Day of month
4. **review_dayofweek**: Day of week effects

### Dropped Features

- `review_id`: Not predictive
- `review_text`: Text analysis not implemented (future work)

## Preprocessing Pipeline

### 1. Label Encoding
All categorical features are label encoded:
```python
LabelEncoder().fit_transform(df[categorical_col])
```

### 2. Standard Scaling
Numerical features are standardized:
```python
StandardScaler().fit_transform(df[numerical_cols])
```

### 3. Type Conversion
All features converted to `float64` for XGBoost compatibility

## Feature Importance

Based on typical model training, the most important features are:

1. **rating** (★★★★★) - Strongest predictor
2. **battery_life_rating** (★★★★)
3. **camera_rating** (★★★★)
4. **performance_rating** (★★★★)
5. **display_rating** (★★★★)
6. **design_rating** (★★★)
7. **price_usd** (★★★)
8. **brand** (★★★)
9. **model** (★★★)
10. **helpful_votes** (★★)

## Data Quality

### Missing Values
- **Count**: 0 (no missing values in dataset)
- **Strategy**: Not applicable

### Outliers
- **Price**: Some luxury models >$1500
- **Age**: Realistic range 18-60
- **Ratings**: All within valid 1-5 range

### Class Imbalance
- **Positive**: ~60% (majority class)
- **Negative**: ~20% (minority)
- **Neutral**: ~20% (minority)
- **Handling**: Stratified train-test split, class weights in future iterations

## Usage Recommendations

### For Training
- Use stratified sampling to maintain class distribution
- Consider SMOTE for balancing if needed
- Test-train split: 80-20
- Cross-validation: 5-fold stratified

### For Inference
- Ensure all 23 required features are present
- Dates should be in YYYY-MM-DD format
- Ratings should be integers 1-5
- Boolean fields should be True/False
- Categorical values should match training vocabulary

### For Feature Engineering
- **Future Work**:
  - Add TF-IDF features from `review_text`
  - Add sentiment scores from text analysis
  - Engineer interaction features (price × rating)
  - Add brand reputation scores
  - Include competitor comparison features

## Example Records

### Positive Sentiment Example
```python
{
    "age": 28,
    "brand": "Apple",
    "model": "iPhone 14",
    "price_usd": 999,
    "rating": 5,
    "battery_life_rating": 5,
    "camera_rating": 5,
    "performance_rating": 5,
    "design_rating": 5,
    "display_rating": 4,
    "verified_purchase": True,
    "helpful_votes": 10,
    "sentiment": "Positive"
}
```

### Negative Sentiment Example
```python
{
    "age": 45,
    "brand": "Samsung",
    "model": "Galaxy A55",
    "price_usd": 350,
    "rating": 1,
    "battery_life_rating": 1,
    "camera_rating": 2,
    "performance_rating": 1,
    "design_rating": 2,
    "display_rating": 2,
    "verified_purchase": True,
    "helpful_votes": 5,
    "sentiment": "Negative"
}
```

### Neutral Sentiment Example
```python
{
    "age": 35,
    "brand": "Xiaomi",
    "model": "Redmi Note 13",
    "price_usd": 450,
    "rating": 3,
    "battery_life_rating": 3,
    "camera_rating": 3,
    "performance_rating": 3,
    "design_rating": 3,
    "display_rating": 3,
    "verified_purchase": True,
    "helpful_votes": 2,
    "sentiment": "Neutral"
}
```

## Dataset Biases

### Known Biases
1. **Platform Bias**: Amazon reviews tend to be more positive
2. **Brand Bias**: Premium brands (Apple) have higher baseline ratings
3. **Verification Bias**: Verified purchases may have different sentiment distribution
4. **Temporal Bias**: Recent reviews may differ from historical trends
5. **Regional Bias**: Cultural differences in review expression

### Mitigation Strategies
- Include `source` as a feature
- Use `verified_purchase` as a feature
- Include temporal features
- Consider country/language context
- Monitor model performance across subgroups

## Data Ethics

- **Privacy**: Customer names are anonymized/pseudonymized
- **Consent**: Reviews assumed to be publicly posted
- **Fairness**: Monitor for bias across demographics
- **Transparency**: Document all preprocessing steps

## References

- Original dataset: Mobile_Reviews_Sentiment.csv
- Records: 50,000
- Features: 25 (24 used for modeling)
- Target: 3-class sentiment

---

**Last Updated**: December 2025
