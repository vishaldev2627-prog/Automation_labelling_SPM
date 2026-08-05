# The official MLflow server image doesn't bundle a Postgres driver or an S3
# client - it's meant to be paired with whatever backend/artifact store you
# use. Extending it here rather than installing at container start (a
# `command: pip install ... && mlflow server ...` one-liner) so a
# transient PyPI outage can't turn a routine restart into a failed one.
FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.9 boto3==1.35.36
