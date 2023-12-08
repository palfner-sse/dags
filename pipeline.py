import astro
from airflow.decorators import dag, task
from pendulum import datetime
from astro.dataframes.pandas import DataFrame
from mlflow_provider.hooks.client import MLflowClientHook
from mlflow_provider.operators.registry import CreateRegisteredModelOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator

## MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
MINIO_CONN_ID = "minio_local"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 100
EXPERIMENT_NAME = "energy"
REGISTERED_MODEL_NAME = "energy_model"
ARTIFACT_BUCKET = "mlflowdataenergy"

@dag(
    schedule=None,
    start_date=datetime(2023, 12, 8),
    catchup=False
)

def pipeline():
    create_buckets_if_not_exists = S3CreateBucketOperator(
        task_id="create_buckets_if_not_exists",
        aws_conn_id=MLFLOW_CONN_ID,
        bucket_name=ARTIFACT_BUCKET,
    )

    @task
    def create_experiment(experiment_name, artifact_bucket, **context):
        ts = context["ts"]

        mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        new_experiment_information = mlflow_hook.run(
            endpoint="api/2.0/mlflow/experiments/create",
            request_params={
                "name": ts + experiment_name,
                "artifact_location": f"s3://{artifact_bucket}/"
            }
        ).json()

        return new_experiment_information["experiment_id"]

    @task
    def scale_features(experiment_id: str) -> astro.dataframes.pandas.DataFrame:
        """Track feature scaling by sklearn in Mlflow."""
        from sklearn.datasets import fetch_california_housing
        from sklearn.preprocessing import StandardScaler
        import mlflow
        import pandas as pd

        df = fetch_california_housing(download_if_missing=True, as_frame=True).frame

        mlflow.sklearn.autolog()

        target = "MedHouseVal"
        X = df.drop(target, axis=1)
        y = df[target]

        scaler = StandardScaler()

        with mlflow.start_run(experiment_id=experiment_id, run_name="Scaler") as run:
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            mlflow.log_metrics(pd.DataFrame(scaler.mean_, index=X.columns)[0].to_dict())

        X[target] = y

    create_registered_model = CreateRegisteredModelOperator(
        task_id="create_registered_model",
        name="{{ ts }}" + "_" + REGISTERED_MODEL_NAME,
        tags=[
            {"key": "model_type", "value": "regression"},
            {"key": "data", "value": "housing"},
        ],
    )

    experiment_created = create_experiment(
        experiment_name=EXPERIMENT_NAME, artifact_bucket=ARTIFACT_BUCKET
    )

    (
        create_buckets_if_not_exists
        >> experiment_created
        >> scale_features(experiment_id=experiment_created)
        >> create_registered_model,
    )


pipeline()

