from __future__ import annotations

from pathlib import Path

import pytest

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder


test_dir: Path = Path(__file__).parent.resolve()


testdata = (
    ("Амурский лесной кот", "Амурский лесной кот"),
    ("Аргали", "Аргали"),
    ("Барсук", "Барсук"),
    ("Бурый медведь", "Бурый медведь"),
    ("Волк", "Волк"),
    ("Гималайский медведь", "Гималайский медведь"),
    ("Енотовидная собака", "Енотовидная собака"),
    ("Заяц", "Заяц"),
    ("Изюбрь", "Изюбрь"),
    ("Ирбис", "Ирбис"),
    ("Кабан", "Кабан"),
    ("Кабарга", "Кабарга"),
    ("Козерог", "Козерог"),
    ("Леопард", "Леопард"),
    ("Лиса", "Лиса"),
    ("Манул", "Манул"),
    ("Марал", "Марал"),
    ("Пятнистый олень", "Пятнистый олень"),
    ("Росомаха", "Росомаха"),
    ("Рысь", "Рысь"),
    ("Сибирская косуля", "Сибирская косуля"),
    ("Соболь", "Соболь"),
    ("Сурок", "Сурок"),
    ("Тигр", "Тигр"),
    ("Харза", "Харза"),
    ("Другие животные", "Другие животные"),
)

@pytest.mark.parametrize("label,expected_unified_label", testdata)
def test_unification_mapping_supported_classes_only(
    label: str,
    expected_unified_label: str | None,
):
    
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_unified_label(label) == expected_unified_label


testdata = (
    ("Азиатский барсук", "Барсук"),
    ("Барсук", "Барсук"),
    ("Гималайский медведь", "Гималайский медведь"),
    ("Черный медведь", "Гималайский медведь"),
    ("Медведь", "Бурый медведь"),
    ("Бурый медведь", "Бурый медведь"),
    ("Снежный барс", "Ирбис"),
    ("Ирбис", "Ирбис"),
    ("Леопард", "Леопард"),
    ("Амурский леопард", "Леопард"),
    ("Тигр", "Тигр"),
    ("Амурский тигр", "Тигр"),
    ("Колонок", "Другие животные"),
    ("Корова", "Другие животные"),
    ("Собака", "Другие животные"),
)

@pytest.mark.parametrize("label,expected_unified_label", testdata)
def test_unification_mapping_classes_to_unify_only(
    label: str,
    expected_unified_label: str | None,
):
    
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_unified_label(label) == expected_unified_label


testdata = (
    ("Золотая рабка", None),
    ("Кот", None),
    ("Канарейке", None),
    ("Коза", None),
)

@pytest.mark.parametrize("label,expected_unified_label", testdata)
def test_unification_mapping_unsopported_classes_only(
    label: str,
    expected_unified_label: str | None,
):
    
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_unified_label(label) == expected_unified_label


testdata = (
    ("Другие животные", 0),
    ("Амурский лесной кот", 1),
    ("Аргали", 2),
    ("Барсук", 3),
    ("Бурый медведь", 4),
    ("Волк", 5),
    ("Гималайский медведь", 6),
    ("Енотовидная собака", 7),
    ("Заяц", 8),
    ("Изюбрь", 9),
    ("Ирбис", 10),
    ("Кабан", 11),
    ("Кабарга", 12),
    ("Козерог", 13),
    ("Леопард", 14),
    ("Лиса", 15),
    ("Манул", 16),
    ("Марал", 17),
    ("Пятнистый олень", 18),
    ("Росомаха", 19),
    ("Рысь", 20),
    ("Сибирская косуля", 21),
    ("Соболь", 22),
    ("Сурок", 23),
    ("Тигр", 24),
    ("Харза", 25),
)

@pytest.mark.parametrize("label,expected_index", testdata)
def test_label_to_index_supported_classes_only(
    label: str,
    expected_index: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_index_by_label(label) == expected_index


testdata = (
    ("Азиатский барсук", 3),
    ("Барсук", 3),
    ("Медведь", 4),
    ("Бурый медведь", 4),
    ("Гималайский медведь", 6),
    ("Черный медведь", 6),
    ("Ирбис", 10),
    ("Снежный барс", 10),
    ("Леопард", 14),
    ("Амурский леопард", 14),
    ("Тигр", 24),
    ("Амурский тигр", 24),
    ("Колонок", 0),
    ("Корова", 0),
    ("Собака", 0),
)

@pytest.mark.parametrize("label,expected_index", testdata)
def test_label_to_index_classes_to_unify_only(
    label: str,
    expected_index: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_index_by_label(label) == expected_index


testdata = (
    ("Слон", None),
    ("Оцелот", None),
    ("Обезьяна", None),
    ("Бык", None),
    ("Крот", None),
)

@pytest.mark.parametrize("label,expected_index", testdata)
def test_label_to_index_upsupported_classes_only(
    label: str,
    expected_index: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_index_by_label(label) == expected_index


testdata = (
    (0, "Other animals"),
    (1, "Leopard cat"),
    (2, "Argali"),
    (3, "European badger"),
    (4, "Brown bear"),
    (5, "Wolf"),
    (6, "Asian black bear"),
    (7, "Raccoon dog"),
    (8, "Hare"),
    (9, "Manchurian wapiti"),
    (10, "Snow leopard"),
    (11, "Wild boar"),
    (12, "Siberian musk deer"),
    (13, "Alpine ibex"),
    (14, "Leopard"),
    (15, "Fox"),
    (16, "Pallas's cat"),
    (17, "Altai wapiti"),
    (18, "Roe deer"),
    (19, "Wolverine"),
    (20, "Lynx"),
    (21, "Siberian roe deer"),
    (22, "Sable"),
    (23, "Ground squirrel"),
    (24, "Tiger"),
    (25, "Yellow-throated marten"),
)

@pytest.mark.parametrize("index,english_label", testdata)
def test_label_to_index_upsupported_classes_only(
    index: int,
    english_label: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
        test_dir / 'assets' / 'test_label_encoder' / 'russian_to_english_mapping.json',
    )

    assert label_encoder.get_english_label_by_index(index) == english_label
