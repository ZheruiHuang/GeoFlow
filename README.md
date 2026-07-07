# GeoFlow

Official code for [GeoFlow: Geo-Aware Modeling of Inter-Area Relationships in Origin-Destination Flow Prediction and Generation](https://arxiv.org/abs/2607.05257), which is accepted by The Forty-Third International Conference on Machine Learning (ICML 2026).

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the raw inputs in these paths:

```text
raw/commuting/data/
raw/commuting/assets/Boundaries_Regions_within_Areas/
raw/FAF/FAF5.7.1_2018-2024.zip
raw/FAF/FAF5_metadata.xlsx
raw/tourism/Flows_visitor_days_origin_fixed.xlsx
```

Then build the processed datasets:

```bash
bash scripts/prepare_all_data.sh
```

## Run

Using the commuting dataset as an example, running prediction experiment:

```bash
python GeoFlow/main.py \
  --dataset-root datasets/commuting \
  --prediction \
  --device cuda:0
```

and generation experiment:

```bash
python GeoFlow/main.py \
  --dataset-root datasets/commuting \
  --generation \
  --device cuda:0
```

More configurations can be specified in `main.py`.