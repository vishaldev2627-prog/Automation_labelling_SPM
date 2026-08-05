-- Postgres only runs files under /docker-entrypoint-initdb.d on first init
-- of an *empty* data directory - so this fires automatically for a brand
-- new deployment, but not against the already-initialized production
-- volume. There, run the single CREATE DATABASE statement below manually
-- once (it's additive and non-destructive - nothing reads or writes this
-- database until the mlflow service's backend-store-uri points at it).
--
-- Separate database, not a schema inside `annotator` - MLflow owns its own
-- Alembic-managed migrations for this database and must never be able to
-- collide with this application's own tables. Same Postgres server/role
-- (POSTGRES_USER/PASSWORD), just a second database on it - no new secret
-- to manage.
SELECT 'CREATE DATABASE mlflow_tracking'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_tracking')\gexec
