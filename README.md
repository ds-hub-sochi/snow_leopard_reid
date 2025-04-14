# CLI

```bash
python ./irbis_classifier/cli/preprocessing/find_series.py \
    --path_to_data_dir ./data/raw/full_images \
    --path_to_save_dir ./data/interim/stage_with_series \
    --old_stages "1,2,3,4,5,6" \
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
nohup python ./irbis_classifier/cli/training/start_training.py --path_to_config ./data/configs/training_config.json &
```

```bash
python ./irbis_classifier/cli/saving/save_as_traced_model.py --path_to_config ./data/configs/saving_config.json 
```

```bash
python ./irbis_classifier/cli/testing/start_testing.py --path_to_config ./data/configs/testing_config.json
```

```bash
python ./irbis_classifier/cli/synthetic_data_gereration/generate_examples.py  \
    --path_to_config ./data/configs/generation_config.json \
    --dump_dir ./data/flux_examples
```

```bash
nohup python ./irbis_classifier/cli/synthetic_data_gereration/extend_train_dataset.py  \
    --path_to_config ./data/configs/generation_config.json \
    --path_to_unification_mapping_json ./data/configs/unification_mapping.json \
    --path_to_supported_labels_json ./data/configs/supported_classes.json \
    --path_to_russian_to_english_mapping_json ./data/configs/russian_to_english_mapping.json &
```

# Выброшенные классы

For 14.04.25:

* выдра - используются только данные с Калужских засек (stage 5)
* косуля (в паке с Калужских засек используется в объединенном лейбле с Сибирской Косулей)
* бенгальская кошка

# Смысл некоторых конфигов

* **unification_mapping** - у каждой картинки уже есть лейбл - они буквально лежат в папках, сформированных по лейблам. Здесь порядок не важен, потому что это словарик.
Этот маппинг делает отображение из таких лейблов в некоторые более обширные классы. Например, 
    * снежный барс -> ибрис;
    * ирбис -> ирбис;

* **supported_classes** - это те классы, которые мы в итоге используем. Именно по этому списку:
    * формируется отображение из лейбла в индекс и обратно. Поэтому важно дописывать новые классы в конец, если мы хотим сохранить порядок логитов на выходе из модели. Если мы переименовываем класс, то важно оставить его на старом месте. Наприме, при добавлении stage_5 было важно было заменить 'Сибирскую косулю' на 'Косулю' именно с помощью преименования, оставляя лейбл на месте, чтобы сохранить порядок логитов;
    * фильтруются классы, которые не будут использоваться. Исключаются все классы, которых нет в этом списке;

* **russian_to_english_mapping** - отображение из названий на русском в названия на английском.

# Про ветки

* develop - основная векта для разработки; время от времени законченные изменения вмерживаются в develop вектку
* main - основная векта репозитория, у нас она используется скорее как релизная ветка. После какого-то значимого изменения develop вмерживается в main;
* ветки с точечными изменениями, например, добавлением нового датасета или новой фичи, должна называться в соответсвии с шаблоном 
**(feat|data|hotfix) \ (дата создания ветки) - (краткое описания изменения)**. Например, feat\14-04-25-add_timm_integration

**Важно**: в develop и main нельзя пушить. Это сделано для того, чтобы в эти наиболее важные ветки уходили только проверенные изменения. Чтобы актуализировать эти ветки, необходимо сделать пул реквест сначала в develop, а затем, когда изменений будет достаточно, из develop в main. В качестве ревьювера нужно указать одного из @windowsartes, @izemtsova, @Ladanovasv
