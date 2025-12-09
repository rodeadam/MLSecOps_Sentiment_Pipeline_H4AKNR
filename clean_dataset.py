"""
Clean the Mobile Reviews dataset by removing currency symbols
"""
import pandas as pd
import re

# Read the dataset
df = pd.read_csv('data/Mobile_Reviews_Sentiment.csv')

print(f"Original dataset shape: {df.shape}")

# Clean price_local column - remove all currency symbols
df['price_local'] = df['price_local'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['price_local'] = pd.to_numeric(df['price_local'], errors='coerce')

# Fill any NaN values with 0 or drop them
df['price_local'] = df['price_local'].fillna(0)

print(f"Cleaned dataset shape: {df.shape}")
print(f"Sample price_local values: {df['price_local'].head()}")

# Save cleaned dataset
df.to_csv('data/Mobile_Reviews_Sentiment_cleaned.csv', index=False)
print("Cleaned dataset saved to: data/Mobile_Reviews_Sentiment_cleaned.csv")
