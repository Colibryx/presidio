#!/bin/sh
exec poetry run gunicorn -w "$WORKERS" -b "0.0.0.0:$PORT" --timeout 300 "app:create_app()"
