from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from mlflow_provider.hooks.client import MLflowClientHook
from mlflow_provider.operators.registry import CreateRegisteredModelOperator
from pendulum import datetime

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
        aws_conn_id=MINIO_CONN_ID,
        bucket_name=ARTIFACT_BUCKET,
    )

    @task
    def create_experiment(experiment_name, artifact_bucket, **context):
        """Create a new MLFlow experiment with a specified name.
        Save artifacts to the specified S3 bucket."""

        ts = context["ts"]

        mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        new_experiment_information = mlflow_hook.run(
            endpoint="api/2.0/mlflow/experiments/create",
            request_params={
                "name": ts + "_" + experiment_name,
                "artifact_location": f"s3://{artifact_bucket}/",
            },
        ).json()

        return new_experiment_information["experiment_id"]

    @task
    def trainModel(experiment_name, artifact_bucket):
        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Load the Iris dataset
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

        # Train a RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

        # Log the model with MLflow
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_param("n_estimators", 10)

            # Log the sklearn model
            mlflow.sklearn.log_model(model, "model")

            # Log model metrics
            mlflow.log_metric("accuracy", accuracy)

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
        >> create_registered_model,
    )


pipeline()
