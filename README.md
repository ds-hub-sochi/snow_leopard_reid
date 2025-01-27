# CLI usage

```bash
python ./irbis_classifier/cli/preprocessing/find_series.py \
    --path_to_data_dir ./data/raw/full_images \
    --path_to_save_dir ./data/interim/stage_with_series
```

```bash
python ./irbis_classifier/cli/preprocessing/filter_duplicates.py \
    --path_to_data_dir ./data/interim/stage_with_series \
    --path_to_save_dir ./data/interim/stage_with_series_filtered
```

```bash
python ./irbis_classifier/cli/preprocessing/train_val_test_split.py \
    --train_size 0.6 \
    --val_size 0.2 \
    --path_to_dir_with_stages ./data/interim/stage_with_series_filtered \
    --path_to_markup_dir ./data/raw/detection_labels \
    --path_to_save_dir ./data/interim/train_val_test_split  
```

```bash
python ./irbis_classifier/cli/preprocessing/match_splits_with_markup.py \
    --path_to_dir_with_splits ./data/interim/train_val_test_split \
    --path_to_markup_dir ./data/raw/detection_labels \
    --path_to_save_dir ./data/processed \
    --min_relative_size 0.02
```

```bash
python ./irbis_classifier/cli/reports/dataset_statistics.py \
    --path_to_data_dir ./data/processed \
    --path_to_save_dir ./reports/figures
```