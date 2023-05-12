#!/bin/sh

sops --encrypt -in-place --pgp $PGP_PUBLIC_KEY database/config.json
sops --encrypt --in-place --pgp $PGP_PUBLIC_KEY tests/fixtures/db_config.json
