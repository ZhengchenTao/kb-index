The deployment pipeline builds one container image per commit on the main branch and promotes the same
image from staging to production. Nothing is rebuilt between environments.

# Deployment

## Pipeline

1. CI runs the unit tests and builds the image `webapp:<short-sha>`.
2. The image is pushed to the registry and deployed to staging automatically.
3. A maintainer approves the production rollout in the CI interface.

The application listens on port 8081 inside the container; the reverse proxy terminates TLS and forwards
to it. Health checks call `/healthz` every ten seconds.

## Rollback

Redeploy the previous image tag. Database migrations are written to be backwards compatible for one
release, so rolling back the application never requires rolling back the schema. See [[backups]] for the
rare case where data has to be restored.

## Manual rollout over SSH (deprecated)

Before the pipeline existed, releases were copied to the server with rsync and restarted by hand.
This procedure is kept for reference only; the full write-up is in archive/legacy-deploy.md.
