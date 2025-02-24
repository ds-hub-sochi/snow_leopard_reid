# CLI usage

```bash
python ./irbis_classifier/cli/preprocessing/find_series.py \
    --path_to_data_dir ./data/raw/full_images \
    --path_to_save_dir ./data/interim/stage_with_series \
    --path_to_unification_mapping_json ./data/configs/unification_mapping.json \
    --path_to_supported_labels_json ./data/configs/supported_classes.json \
    --path_to_russian_to_english_mapping_json ./data/configs/russian_to_english_mapping.json
```

```bash
python ./irbis_classifier/cli/preprocessing/filter_broken_images.py \
    --path_to_data_dir ./data/interim/stage_with_series \
    --path_to_save_dir ./data/interim/stage_with_series_filtered
```

```bash
python ./irbis_classifier/cli/preprocessing/filter_duplicates.py \
    --path_to_data_dir ./data/interim/stage_with_series_filtered \
    --path_to_save_dir ./data/interim/stage_with_series_without_duplicates
```

```bash
python ./irbis_classifier/cli/preprocessing/sample_from_long_series.py \
    --path_to_data_dir ./data/interim/stage_with_series_without_duplicates \
    --path_to_save_dir ./data/interim/stage_with_resampled_series \
    --classes_to_sample_json ./data/configs/classes_to_sample.json \
    --max_sequence_length 40 \
    --resample_size 0.25
```

```bash
python ./irbis_classifier/cli/reports/sequence_lenght.py \
    --path_to_data_dir_before ./data/interim/stage_with_series_without_duplicates \
    --path_to_data_dir_after ./data/interim/stage_with_resampled_series \
    --path_to_save_dir ./reports/figures \
    --max_sequence_length 40
```

```bash
python ./irbis_classifier/cli/preprocessing/train_val_test_split.py \
    --train_size 0.6 \
    --val_size 0.2 \
    --path_to_dir_with_stages ./data/interim/stage_with_resampled_series \
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

```bash
python ./irbis_classifier/cli/reports/test_augmentations.py \
    --path_to_data_file ./data/processed/train.csv \
    --path_to_save_dir ./reports/figures \
    --n_samples 35
```

```bash
nohup python ./irbis_classifier/cli/training/start_training.py \
    --path_to_data_dir ./data/processed \
    --path_to_checkpoints_dir ./models \
    --path_to_experiment_config ./data/configs/experiment.json \
    --model_name EfficientNet_V2_L \
    --run_name EfficientNet_V2_L_CrossEntropyLoss_smoothing_scheduler_warmup \
    --batch_size 512 \
    --n_epochs 30 \
    --lr 1e-4 \
    --device_ids "0,1" \
    --path_to_unification_mapping_json ./data/configs/unification_mapping.json \
    --path_to_supported_labels_json ./data/configs/supported_classes.json \
    --path_to_russian_to_english_mapping_json ./data/configs/russian_to_english_mapping.json \
    --use_scheduler True \
    --warmup_epochs 5 \
    --use_weighted_loss False \
    --loss CrossEntropyLoss \
    --label_smoothing 0.05 &
```

```bash
python ./irbis_classifier/cli/testing/start_testing.py \
    --path_to_test_csv ./data/processed/test.csv \
    --model_name Swin_B \
    --path_to_weight ./models/Swin_B_CrossEntropyLoss_smoothing_scheduler_warmup/2025-02-21/best_model.pth \
    --batch_size 256 \
    --bootstrap_size 100000 \
    --alpha 0.95 \
    --path_to_save_dir ./reports/figures \
    --path_to_unification_mapping_json ./data/configs/unification_mapping.json \
    --path_to_supported_labels_json ./data/configs/supported_classes.json \
    --path_to_russian_to_english_mapping_json ./data/configs/russian_to_english_mapping.json
```

```bash
python ./irbis_classifier/cli/saving/save_as_traced_model.py \
    --model_name Swin_B \
    --path_to_weight ./models/Swin_B_CrossEntropyLoss_smoothing_scheduler_warmup/2025-02-21/best_model.pth \
    --path_to_unification_mapping_json ./data/configs/unification_mapping.json \
    --path_to_supported_labels_json ./data/configs/supported_classes.json \
    --path_to_russian_to_english_mapping_json ./data/configs/russian_to_english_mapping.json \
    --path_to_traced_model_checkpoint ./models/jit/Swin_B.pt
```

# Dropped labels:

For 28.01.25:

* выдра
* косуля
* бенгальская кошка
* хорек
