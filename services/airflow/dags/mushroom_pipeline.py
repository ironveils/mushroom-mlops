from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/project"


with DAG(
    dag_id="mushroom_mlops_pipeline",
    start_date=datetime(2026, 9, 16),
    schedule="*/5 * * * *",
    catchup=False,
    tags=["mushroom", "mlops"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python code/datasets/preprocess.py"
        ),
    )

    train = BashOperator(
        task_id="train_and_evaluate",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python code/models/train.py"
        ),
    )

    deploy = BashOperator(
        task_id="deploy",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "docker compose -f code/deployment/docker-compose.yml "
            "up -d --build"
        ),
    )

    preprocess >> train >> deploy