# D1 - Orquestrador Airflow
FROM apache/airflow:2.10.1-python3.11

USER root
RUN apt-get update && apt-get install -y docker-cli git

USER airflow
WORKDIR /opt/airflow
COPY dags/ ./dags/
