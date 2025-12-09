"""
Flask REST API for Mobile Reviews Sentiment Analysis
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import pandas as pd
import mlflow
import os
from MLModel import SentimentMLModel
from constants import *

# Initialize Flask app
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Mobile Sentiment Analysis API',
    description='REST API for training and predicting mobile review sentiment',
    doc='/'
)

# Namespaces
ns_model = api.namespace('model', description='Model operations')

# Models for request/response documentation
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True, help='CSV file with training data')

predict_model = api.model('PredictRequest', {
    'reviews': fields.List(fields.Raw, required=True, description='List of review dictionaries')
})

register_model = api.model('RegisterRequest', {
    'run_id': fields.String(required=True, description='MLflow run ID'),
    'model_name': fields.String(required=True, description='Model name for registry')
})

# Global variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
current_model = None
staging_run_id = None


def load_staging_model():
    """Load the current staging model from MLflow"""
    global current_model, staging_run_id
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Get all runs from the default experiment
        runs = client.search_runs(
            experiment_ids=['0'],
            filter_string="tags.stage='Staging'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            staging_run_id = run.info.run_id
            
            # Load model
            current_model = SentimentMLModel()
            current_model.load_model_from_mlflow(staging_run_id, MLFLOW_TRACKING_URI)
            
            print(f"Loaded staging model from run: {staging_run_id}")
            print(f"Test accuracy: {run.data.metrics.get('test_accuracy', 'N/A')}")
        else:
            print("No staging model found. Train a model first.")
            current_model = None
            
    except Exception as e:
        print(f"Error loading staging model: {e}")
        current_model = None


@ns_model.route('/train')
class TrainModel(Resource):
    @api.expect(upload_parser)
    @api.doc(responses={
        200: 'Success',
        400: 'Invalid input',
        500: 'Internal server error'
    })
    def post(self):
        """Train a new sentiment analysis model with CSV data"""
        try:
            # Check if file is present
            if 'file' not in request.files:
                return {'message': 'No file provided'}, 400
            
            file = request.files['file']
            
            if file.filename == '':
                return {'message': 'Empty filename'}, 400
            
            # Read CSV
            df = pd.read_csv(file)
            
            # Validate required columns
            required_cols = [TARGET_COLUMN] + ALL_FEATURES
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    'message': f'Missing required columns: {missing_cols}'
                }, 400
            
            # Train model
            model = SentimentMLModel()
            results = model.train(df, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
            
            # Tag as staging
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            client.set_tag(results['run_id'], "stage", "Staging")
            
            # Reload staging model
            load_staging_model()
            
            return {
                'message': 'Model trained successfully!',
                'run_id': results['run_id'],
                'train_accuracy': results['train_accuracy'],
                'test_accuracy': results['test_accuracy'],
                'classification_report': results['classification_report']
            }, 200
            
        except Exception as e:
            return {'message': f'Training failed: {str(e)}'}, 500


@ns_model.route('/predict')
class PredictSentiment(Resource):
    @api.expect(predict_model)
    @api.doc(responses={
        200: 'Success',
        400: 'Invalid input',
        500: 'Internal server error'
    })
    def post(self):
        """Make sentiment predictions on new reviews"""
        try:
            # Load staging model if not loaded
            if current_model is None:
                load_staging_model()
            
            if current_model is None:
                return {
                    'message': 'No staging model loaded. Train a model first.'
                }, 400
            
            # Get request data
            data = request.get_json()
            
            if 'reviews' not in data:
                return {'message': 'No reviews provided'}, 400
            
            # Create dataframe from reviews
            df = pd.DataFrame(data['reviews'])
            
            # Make predictions
            predictions = current_model.predict(df)
            
            return {
                'predictions': predictions.tolist(),
                'message': 'Predictions made successfully'
            }, 200
            
        except Exception as e:
            return {'message': f'Prediction failed: {str(e)}'}, 500


@ns_model.route('/register')
class RegisterModel(Resource):
    @api.expect(register_model)
    @api.doc(responses={
        200: 'Success',
        400: 'Invalid input',
        500: 'Internal server error'
    })
    def post(self):
        """Register a trained model in MLflow Model Registry"""
        try:
            data = request.get_json()
            
            run_id = data.get('run_id')
            model_name = data.get('model_name', 'Mobile_Sentiment_Model')
            
            if not run_id:
                return {'message': 'run_id is required'}, 400
            
            # Register model
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"runs:/{run_id}/model"
            
            mlflow.register_model(model_uri, model_name)
            
            return {
                'message': f'Model registered successfully as {model_name}',
                'run_id': run_id
            }, 200
            
        except Exception as e:
            return {'message': f'Registration failed: {str(e)}'}, 500


@ns_model.route('/status')
class ModelStatus(Resource):
    @api.doc(responses={200: 'Success'})
    def get(self):
        """Get current model status"""
        global current_model, staging_run_id
        
        if current_model is None:
            load_staging_model()
        
        return {
            'staging_model_loaded': current_model is not None,
            'staging_run_id': staging_run_id,
            'mlflow_tracking_uri': MLFLOW_TRACKING_URI
        }, 200


# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy'}, 200


# Initialize by loading staging model
with app.app_context():
    load_staging_model()


if __name__ == '__main__':
    # Start MLflow server in a separate process
    import subprocess
    import time
    
    print("Starting MLflow server on port 5102...")
    
    mlflow_process = subprocess.Popen([
        'mlflow', 'server',
        '--host', '0.0.0.0',
        '--port', '5102',
        '--backend-store-uri', MLFLOW_TRACKING_URI,
        '--default-artifact-root', '/app/mlartifacts',
        '--serve-artifacts'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Give MLflow time to start
    time.sleep(2)
    
    # Check if MLflow process started successfully
    if mlflow_process.poll() is not None:
        stdout, stderr = mlflow_process.communicate()
        print(f"ERROR: MLflow server failed to start!")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
    else:
        print("✓ MLflow server started successfully")
    
    print("Starting API server on port 8080...")
    
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=8080, debug=False)
    finally:
        # Clean up MLflow process if Flask app stops
        mlflow_process.terminate()
        mlflow_process.wait(timeout=5)
