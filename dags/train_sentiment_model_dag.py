"""
Airflow DAG for automated sentiment model training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import requests
import mlflow
import os

# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@mlops.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'train_sentiment_model_dag',
    default_args=default_args,
    description='Train mobile sentiment analysis model with promotion workflow',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'sentiment', 'mobile-reviews'],
)


def train_model_task():
    """Train a new sentiment model via API"""
    
    # Path to training data
    data_path = '/opt/airflow/data/Mobile_Reviews_Sentiment_cleaned.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    # Call training API
    api_url = 'http://mlsecops:8080/model/train'
    
    with open(data_path, 'rb') as f:
        files = {'file': ('Mobile_Reviews_Sentiment.csv', f, 'text/csv')}
        response = requests.post(api_url, files=files, timeout=600)
    
    if response.status_code != 200:
        raise Exception(f"Training API failed: {response.text}")
    
    result = response.json()
    print(f"Training completed!")
    print(f"Run ID: {result['run_id']}")
    print(f"Train Accuracy: {result['train_accuracy']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    
    # Store run_id for next task
    return result['run_id']


def compare_and_promote_task(**context):
    """Compare new model with current staging and promote if better"""
    
    # Get new run_id from previous task
    ti = context['ti']
    new_run_id = ti.xcom_pull(task_ids='train_model')
    
    if not new_run_id:
        raise ValueError("No run_id received from training task")
    
    # MLflow setup
    mlflow_uri = 'file:///app/mlruns'
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Get new model metrics
    new_run = client.get_run(new_run_id)
    new_accuracy = new_run.data.metrics.get('test_accuracy', 0)
    
    print(f"New model accuracy: {new_accuracy:.4f}")
    
    # Get current staging model
    staging_runs = client.search_runs(
        experiment_ids=['0'],
        filter_string="tags.stage='Staging'",
        order_by=["metrics.test_accuracy DESC"],
        max_results=1
    )
    
    # Determine if we should promote
    should_promote = True
    promotion_reason = "First model"
    
    if staging_runs and staging_runs[0].info.run_id != new_run_id:
        current_run = staging_runs[0]
        current_accuracy = current_run.data.metrics.get('test_accuracy', 0)
        
        print(f"Current staging model accuracy: {current_accuracy:.4f}")
        
        if new_accuracy > current_accuracy:
            improvement = ((new_accuracy - current_accuracy) / current_accuracy) * 100
            promotion_reason = f"Accuracy improved by {improvement:.2f}%"
            should_promote = True
            
            # Remove staging tag from old model
            client.set_tag(current_run.info.run_id, "stage", "Archived")
        else:
            should_promote = False
            promotion_reason = f"No improvement (current: {current_accuracy:.4f}, new: {new_accuracy:.4f})"
    
    # Promote if appropriate
    if should_promote:
        client.set_tag(new_run_id, "stage", "Staging")
        print(f"✅ Model promoted to Staging! Reason: {promotion_reason}")
        
        # Check if should promote to production (e.g., accuracy > 0.80)
        if new_accuracy > 0.80:
            client.set_tag(new_run_id, "stage", "Production")
            print(f"🚀 Model also promoted to Production! (accuracy > 0.80)")
    else:
        print(f"❌ Model not promoted. Reason: {promotion_reason}")
    
    return {
        'promoted': should_promote,
        'new_accuracy': new_accuracy,
        'reason': promotion_reason
    }


def send_notification_task(**context):
    """Send notification about training results"""
    
    ti = context['ti']
    promotion_result = ti.xcom_pull(task_ids='compare_and_promote')
    
    if not promotion_result:
        print("No promotion result available")
        return
    
    if promotion_result['promoted']:
        message = f"""
        ✅ Sentiment Model Training Successful
        
        New model promoted to Staging!
        Accuracy: {promotion_result['new_accuracy']:.4f}
        Reason: {promotion_result['reason']}
        """
    else:
        message = f"""
        ℹ️ Sentiment Model Training Completed
        
        Model not promoted.
        Accuracy: {promotion_result['new_accuracy']:.4f}
        Reason: {promotion_result['reason']}
        """
    
    print(message)
    # In production, send email or Slack notification here


# Task 1: Train model
train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Task 2: Compare and promote
compare_and_promote = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote_task,
    provide_context=True,
    dag=dag,
)

# Task 3: Send notification
send_notification = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification_task,
    provide_context=True,
    dag=dag,
)

# Task dependencies
train_model >> compare_and_promote >> send_notification
