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
    )

    assert label_encoder.get_unified_label(label) == expected_unified_label


testdata = (
    ("Амурский лесной кот", 0),
    ("Аргали", 1),
    ("Барсук", 2),
    ("Бурый медведь", 3),
    ("Волк", 4),
    ("Гималайский медведь", 5),
    ("Енотовидная собака", 6),
    ("Заяц", 7),
    ("Изюбрь", 8),
    ("Ирбис", 9),
    ("Кабан", 10),
    ("Кабарга", 11),
    ("Козерог", 12),
    ("Леопард", 13),
    ("Лиса", 14),
    ("Манул", 15),
    ("Марал", 16),
    ("Пятнистый олень", 17),
    ("Росомаха", 18),
    ("Рысь", 19),
    ("Сибирская косуля", 20),
    ("Соболь", 21),
    ("Сурок", 22),
    ("Тигр", 23),
    ("Харза", 24),
    ("Другие животные", 25),
)

@pytest.mark.parametrize("label,expected_index", testdata)
def test_label_to_index_supported_classes_only(
    label: str,
    expected_index: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
    )

    assert label_encoder.get_index_by_label(label) == expected_index


testdata = (
    ("Азиатский барсук", 2),
    ("Барсук", 2),
    ("Медведь", 3),
    ("Бурый медведь", 3),
    ("Гималайский медведь", 5),
    ("Черный медведь", 5),
    ("Ирбис", 9),
    ("Снежный барс", 9),
    ("Леопард", 13),
    ("Амурский леопард", 13),
    ("Тигр", 23),
    ("Амурский тигр", 23),
    ("Колонок", 25),
    ("Корова", 25),
    ("Собака", 25),
)

@pytest.mark.parametrize("label,expected_index", testdata)
def test_label_to_index_classes_to_unify_only(
    label: str,
    expected_index: int,
):
    label_encoder: LabelEncoder = create_label_encoder(
        test_dir / 'assets' / 'test_label_encoder' / 'unification_mapping.json',
        test_dir / 'assets' / 'test_label_encoder' / 'supported_classes.json',
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
    )

    assert label_encoder.get_index_by_label(label) == expected_index
