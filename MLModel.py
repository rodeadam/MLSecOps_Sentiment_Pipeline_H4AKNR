"""
Mobile Reviews Sentiment Analysis ML Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.xgboost
import pickle
import os
from datetime import datetime
from constants import *


class SentimentMLModel:
    """
    ML Model for predicting mobile review sentiment using XGBoost
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.sentiment_encoder = LabelEncoder()
        self.feature_names = None
        
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the mobile reviews dataset
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (True) or inference data (False)
            
        Returns:
            Preprocessed features and target (if training)
        """
        df = df.copy()
        
        # Clean currency symbols from price_local column
        if 'price_local' in df.columns:
            df['price_local'] = df['price_local'].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df['price_local'] = pd.to_numeric(df['price_local'], errors='coerce')
        
        # Drop ID column if present
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])
        
        # Drop text features (for now - could add TF-IDF/embeddings later)
        if TEXT_FEATURES[0] in df.columns:
            df = df.drop(columns=TEXT_FEATURES)
        
        # Process date features - extract year, month, day
        if DATE_FEATURES[0] in df.columns:
            df['review_year'] = pd.to_datetime(df[DATE_FEATURES[0]]).dt.year
            df['review_month'] = pd.to_datetime(df[DATE_FEATURES[0]]).dt.month
            df['review_day'] = pd.to_datetime(df[DATE_FEATURES[0]]).dt.day
            df['review_dayofweek'] = pd.to_datetime(df[DATE_FEATURES[0]]).dt.dayofweek
            df = df.drop(columns=DATE_FEATURES)
        
        # Convert binary features to int
        for col in BINARY_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Separate target if training
        if is_training and TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN]
            X = df.drop(columns=[TARGET_COLUMN])
        else:
            X = df
            y = None
        
        # Label encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle unseen categories
                    X[col] = X[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # Add date-derived features to numerical features list
        date_derived_features = ['review_year', 'review_month', 'review_day', 'review_dayofweek']
        features_to_scale = FEATURES_TO_SCALE + date_derived_features
        
        # Scale numerical features
        scale_cols = [col for col in features_to_scale if col in X.columns]
        if is_training:
            X[scale_cols] = self.scaler.fit_transform(X[scale_cols])
        else:
            X[scale_cols] = self.scaler.transform(X[scale_cols])
        
        # Convert all to float
        X = X.astype('float64')
        
        # Store feature names
        if is_training:
            self.feature_names = X.columns.tolist()
        
        # Encode target if training
        if is_training and y is not None:
            y = self.sentiment_encoder.fit_transform(y)
        
        return X, y if is_training else X
    
    def train(self, df, mlflow_tracking_uri=None):
        """
        Train the XGBoost sentiment classifier
        
        Args:
            df: Training dataframe
            mlflow_tracking_uri: MLflow tracking URI
            
        Returns:
            Dictionary with training results
        """
        # Set MLflow tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Preprocess data
            X, y = self.preprocess_data(df, is_training=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            # Log preprocessing info
            mlflow.log_param("preprocessing", "LabelEncoder + StandardScaler")
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("num_samples", X_train.shape[0])
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            self.model.fit(X_train, y_train)
            
            # Log model parameters
            for param, value in XGBOOST_PARAMS.items():
                mlflow.log_param(param, value)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            # Generate classification report
            class_report = classification_report(
                y_test, y_test_pred, 
                target_names=SENTIMENT_CLASSES,
                output_dict=True
            )
            
            # Log per-class metrics
            for sentiment in SENTIMENT_CLASSES:
                mlflow.log_metric(f"{sentiment}_precision", class_report[sentiment]['precision'])
                mlflow.log_metric(f"{sentiment}_recall", class_report[sentiment]['recall'])
                mlflow.log_metric(f"{sentiment}_f1-score", class_report[sentiment]['f1-score'])
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            
            # Save artifacts
            self._save_artifacts()
            
            # Log artifacts to MLflow
            mlflow.xgboost.log_model(self.model, "model")
            mlflow.log_artifact("label_encoders.pkl")
            mlflow.log_artifact("scaler.pkl")
            mlflow.log_artifact("sentiment_encoder.pkl")
            
            # Generate Evidently report
            self._generate_evidently_report(
                X_train, y_train, X_test, y_test, 
                y_train_pred, y_test_pred
            )
            
            print(f"Training completed!")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred, target_names=SENTIMENT_CLASSES)}")
            
            return {
                'run_id': run.info.run_id,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'classification_report': class_report
            }
    
    def predict(self, df):
        """
        Make predictions on new data
        
        Args:
            df: Input dataframe
            
        Returns:
            Predictions as sentiment labels
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        # Preprocess data
        X = self.preprocess_data(df, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Convert back to sentiment labels
        sentiment_labels = self.sentiment_encoder.inverse_transform(predictions)
        
        return sentiment_labels
    
    def _save_artifacts(self):
        """Save preprocessing artifacts"""
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open('sentiment_encoder.pkl', 'wb') as f:
            pickle.dump(self.sentiment_encoder, f)
    
    def load_artifacts(self, artifacts_path='.'):
        """Load preprocessing artifacts"""
        with open(f'{artifacts_path}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f'{artifacts_path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{artifacts_path}/sentiment_encoder.pkl', 'rb') as f:
            self.sentiment_encoder = pickle.load(f)
    
    def _generate_evidently_report(self, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred):
        """Generate Evidently AI report for monitoring"""
        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataQualityPreset
            
            # Create DataFrames with predictions
            train_df = X_train.copy()
            train_df['target'] = y_train
            train_df['prediction'] = y_train_pred
            
            test_df = X_test.copy()
            test_df['target'] = y_test
            test_df['prediction'] = y_test_pred
            
            # Define column mapping
            column_mapping = ColumnMapping(
                target='target',
                prediction='prediction',
                numerical_features=[col for col in NUMERICAL_FEATURES if col in X_test.columns],
                categorical_features=[col for col in CATEGORICAL_FEATURES if col in X_test.columns]
            )
            
            # Create report with DataQualityPreset
            report = Report(metrics=[
                DataQualityPreset()
            ])
            
            report.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping)
            
            # Save report
            report.save_html('/app/reports/report.html')
            print("Evidently report generated successfully!")
            
            # Log to MLflow
            mlflow.log_artifact('/app/reports/report.html')
            
        except Exception as e:
            print(f"Error generating Evidently report: {e}")
    
    def load_model_from_mlflow(self, run_id, mlflow_tracking_uri):
        """
        Load model and artifacts from MLflow
        
        Args:
            run_id: MLflow run ID
            mlflow_tracking_uri: MLflow tracking URI
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load model
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.xgboost.load_model(model_uri)
        
        # Download and load artifacts
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
        self.load_artifacts(artifacts_path)
        
        print(f"Model and artifacts loaded from run: {run_id}")