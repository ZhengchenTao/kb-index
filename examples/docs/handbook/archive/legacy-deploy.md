# Legacy deployment (rsync)

Releases used to be built on a developer laptop, copied to the application server with rsync and
restarted with `systemctl restart webapp`. There was no staging environment and no way to roll back other
than copying the previous build again.

This procedure was replaced by the container pipeline described in [[deploy]].
