"""
Sample script for testing sentiment predictions
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8080/model/predict"

# Sample reviews for prediction
sample_reviews = [
    {
        "customer_name": "John Smith",
        "age": 32,
        "brand": "Apple",
        "model": "iPhone 15 Pro",
        "price_usd": 1199,
        "price_local": 1199,
        "currency": "USD",
        "exchange_rate_to_usd": 1.0,
        "rating": 5,
        "country": "USA",
        "language": "English",
        "review_date": "2025-12-01",
        "verified_purchase": True,
        "battery_life_rating": 5,
        "camera_rating": 5,
        "performance_rating": 5,
        "design_rating": 5,
        "display_rating": 5,
        "review_length": 85,
        "word_count": 18,
        "helpful_votes": 15,
        "source": "Amazon"
    },
    {
        "customer_name": "Jane Doe",
        "age": 28,
        "brand": "Samsung",
        "model": "Galaxy S24",
        "price_usd": 350,
        "price_local": 350,
        "currency": "USD",
        "exchange_rate_to_usd": 1.0,
        "rating": 2,
        "country": "USA",
        "language": "English",
        "review_date": "2025-11-25",
        "verified_purchase": True,
        "battery_life_rating": 2,
        "camera_rating": 2,
        "performance_rating": 1,
        "design_rating": 2,
        "display_rating": 3,
        "review_length": 52,
        "word_count": 9,
        "helpful_votes": 3,
        "source": "BestBuy"
    },
    {
        "customer_name": "Bob Johnson",
        "age": 45,
        "brand": "Google",
        "model": "Pixel 8",
        "price_usd": 699,
        "price_local": 699,
        "currency": "USD",
        "exchange_rate_to_usd": 1.0,
        "rating": 3,
        "country": "USA",
        "language": "English",
        "review_date": "2025-12-05",
        "verified_purchase": False,
        "battery_life_rating": 3,
        "camera_rating": 4,
        "performance_rating": 3,
        "design_rating": 3,
        "display_rating": 3,
        "review_length": 68,
        "word_count": 12,
        "helpful_votes": 1,
        "source": "eBay"
    }
]


def predict_sentiment(reviews):
    """
    Make sentiment predictions using the API
    
    Args:
        reviews: List of review dictionaries
    """
    payload = {"reviews": reviews}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("=" * 60)
        print("SENTIMENT PREDICTIONS")
        print("=" * 60)
        
        for i, (review, prediction) in enumerate(zip(reviews, result['predictions'])):
            print(f"\nReview {i+1}:")
            print(f"  Brand/Model: {review['brand']} {review['model']}")
            print(f"  Rating: {review['rating']}/5")
            print(f"  Price: ${review['price_usd']}")
            print(f"  Predicted Sentiment: {prediction}")
            print("-" * 60)
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is the service running?")
        print("   Start it with: docker-compose -f docker-compose-airflow.yml up -d")
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error: HTTP {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("\n🔮 Making predictions on sample reviews...\n")
    predict_sentiment(sample_reviews)
