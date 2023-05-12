#!/bin/sh

echo "$PGP_PRIVATE_KEY" > private.key || true
echo "$PGP_PUBLIC_KEY" > public.key || true
gpg --import public.key || true
gpg --import private.key || true

sops --decrypt -in-place database/config.json || true
sops --decrypt --in-place tests/fixtures/db_config.json || true
