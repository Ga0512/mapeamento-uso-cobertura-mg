from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'gabriel',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 10, 23),
}

with DAG(
    'ml_train_and_cicd',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
) as dag:

    # 1️⃣ Treina o modelo no host
    train = BashOperator(
        task_id='run_training',
        bash_command='python3 /host_project/pipeline/ train.py',
    )

    # 2️⃣ Dispara o CI/CD (gera D2 no host)
    cicd = BashOperator(
        task_id='run_cicd',
        bash_command='cd /host_project && gh workflow run ci_cd.yml',
    )

    train >> cicd
