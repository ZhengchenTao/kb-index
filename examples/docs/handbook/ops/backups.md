---
status: draft
owner: ops
---

# Backups

## What is backed up

The PostgreSQL database is dumped every night at 02:00 with `pg_dump --format=custom`. Dumps are kept for
fourteen days locally and ninety days in object storage. Uploaded files are synchronised to object storage
every hour.

## Restore

1. Stop the application so nothing writes during the restore.
2. Restore the latest dump with `pg_restore --clean --if-exists` into the production database.
3. Start the application and check `/healthz`.

Restores are rehearsed once per quarter on staging. The deployment side of a restore is described in the
[handbook overview](../README.md) and in [[deploy]].
