from datetime import datetime, timedelta
from io import StringIO

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data_file_path = "/opt/airflow/data/energy_2022"

# Define default_args dictionary to specify the default parameters of the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'energy_data_pipeline',
    default_args=default_args,
    description='A simple machine learning workflow with MLflow',
    schedule_interval=timedelta(days=1),
)

def ingest_train_data(**kwargs):
    file_url = "https://www.entsoe.eu/publications/data/power-stats/2022/monthly_hourly_load_values_2022.csv"

    raw_data_2022 = []

    response = requests.get(file_url)
    if response.status_code == 200:
        content = response.content.decode('utf-8')
        raw_data_2022 = pd.read_csv(StringIO(content), header=0, sep=';', parse_dates=['DateUTC', "DateShort"])
        print("File downloaded and read into DataFrame successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    raw_data_2022 = raw_data_2022.drop("MeasureItem", axis=1)
    raw_data_2022 = raw_data_2022.drop("Cov_ratio", axis=1)
    raw_data_2022 = raw_data_2022.drop("CreateDate", axis=1)

    raw_data_2022["DateUTC"] = pd.to_datetime(raw_data_2022["DateUTC"], format="%d/%m/%Y %H:%M")
    raw_data_2022["DateShort"] = pd.to_datetime(raw_data_2022["DateShort"], format="%d/%m/%Y")

    raw_data_2022["TimeFrom"] = raw_data_2022['TimeFrom'].str.split(':').str[0].astype(float)
    raw_data_2022["TimeTo"] = raw_data_2022['TimeTo'].str.split(':').str[0].astype(float)

    reference_date = pd.to_datetime('2022-01-01')
    raw_data_2022['DateShort'] = (raw_data_2022['DateShort'] - reference_date).dt.days

    country_code_mapping = {
        'AL': 1, 'AT': 2, 'BA': 3, 'BE': 4, 'BG': 5, 'CH': 6, 'CY': 7, 'CZ': 8, 'DE': 9, 'DK': 10, 'EE': 11, 'ES': 12,
        'FI': 13, 'FR': 14, 'GE': 15, 'GR': 16, 'HR': 17, 'HU': 18, 'IE': 19, 'IT': 20, 'LT': 21, 'LU': 22, 'LV': 23,
        'MD': 24, 'ME': 25, 'MK': 26, 'NL': 27, 'NO': 28, 'PL': 29, 'PT': 30, 'RO': 31, 'RS': 32, 'SE': 33, 'SI': 34,
        'SK': 35, 'UA': 36, 'XK': 37
    }

    raw_data_2022['CountryCode'] = raw_data_2022['CountryCode'].map(country_code_mapping)

    raw_data_2022.to_csv(train_data_file_path, index=False)

    target = "Value"
    numerical_features = ["Value_ScaleTo100", "TimeFrom", "TimeTo"]
    categorical_features = ["DateShort", "CountryCode"]
    reference= "'01/01/2022 00:00':'31/6/2022 23:00'"

    kwargs['ti'].xcom_push(key='target', value=target)
    kwargs['ti'].xcom_push(key='numerical_features', value=numerical_features)
    kwargs['ti'].xcom_push(key='categorical_features', value=categorical_features)
    kwargs['ti'].xcom_push(key='reference_data', value=reference)

data_ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=ingest_train_data,
    provide_context=True,
    dag=dag,
)


def train_model(**kwargs):

    ti = kwargs['ti']

    target = ti.xcom_pull(task_ids='ingest_data', key='target')
    numerical_features = ti.xcom_pull(task_ids='ingest_data', key='numerical_features')
    categorical_features = ti.xcom_pull(task_ids='ingest_data', key='categorical_features')
    reference = ti.xcom_pull(task_ids='ingest_data', key='reference_data')

    # Load data
    data = pd.read_csv(train_data_file_path)

    reference = data.loc[reference]

    regressor = LinearRegression()
    regressor.fit(reference[numerical_features + categorical_features], reference[target])

    mlflow.set_tracking_uri("http://10.101.198.233:5000")
    mlflow.set_experiment("Energy")
    run_artifact_path = "mlflow-artifacts:/"
    with mlflow.start_run(run_name="Energy_1") as run:
        mlflow.log_params({"model": "Linear Regression Classifier"})
        mlflow.sklearn.log_model(regressor, "model")
        run_artifact_path = run_artifact_path + mlflow.get_experiment_by_name(
            "Energy").experiment_id + "/" + mlflow.active_run().info.run_id + "/artifacts/model"

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

# Task Dependencies
data_ingestion_task >> model_training_task
