#!/usr/bin/env bash
# Upload each built release to gs://<bucket>/releases/<n>/ for Triton.
# `make models` already writes release vN with Triton version directory N.
set -euo pipefail

models=${1:?usage: publish_triton.sh <models dir> <bucket>}
bucket=${2:?usage: publish_triton.sh <models dir> <bucket>}

for release in "$models"/v*; do
  n=${release##*/v}
  echo "publishing $release -> gs://$bucket/releases/$n/"
  gcloud storage rsync --recursive --delete-unmatched-destination-objects \
    --exclude 'model_card\.json$' "$release" "gs://$bucket/releases/$n"
done
