from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_dag',
    default_args=default_args,
    description='A simple DAG for model training',
    schedule_interval=timedelta(days=1),
)

def download_dataset():
    url = "https://www.entsoe.eu/publications/data/power-stats/2022/monthly_hourly_load_values_2022.csv"
    df = pd.read_csv(url)
    df.to_csv('/path/to/save/dataset.csv', index=False)

def preprocess_data():
    # Add data preprocessing steps here
    pass

def train_model():
    # Load dataset
    df = pd.read_csv('/path/to/save/dataset.csv')

    # Example: Predicting some target based on features
    X = df[['feature1', 'feature2']]  # Replace with actual features
    y = df['target']  # Replace with actual target

    model = RandomForestRegressor()
    model.fit(X, y)

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

download_task >> preprocess_task >> train_model_task
