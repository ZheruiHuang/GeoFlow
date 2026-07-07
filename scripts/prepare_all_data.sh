#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

rm -rf datasets/commuting datasets/faf datasets/airtravel
mkdir -p datasets

"${PYTHON}" data_prep/preprocess_commuting.py \
  --raw-data-dir raw/commuting/data \
  --asset-dir raw/commuting/assets/Boundaries_Regions_within_Areas \
  --out-root datasets/commuting

"${PYTHON}" data_prep/build_faf_dataset.py \
  --faf-csv raw/FAF/FAF5.7.1_2018-2024.zip \
  --metadata raw/FAF/FAF5_metadata.xlsx \
  --out-root datasets/faf

"${PYTHON}" data_prep/build_airtravel_dataset.py \
  --src raw/tourism/Flows_visitor_days_origin_fixed.xlsx \
  --out-root datasets/airtravel \
  --augment-subsamples 21 \
  --split-shuffle-seed 13 \
  --split-seeds 0
