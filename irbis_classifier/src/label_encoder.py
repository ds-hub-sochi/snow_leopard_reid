"""This is a module for easy manipulating with labels"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelEncoder:
    _unification_mapping: dict[str, str]
    _supported_labels: tuple[str]
    _russian_to_english_mapping: dict[str, str]
    _label_to_index: dict[str, int] = field(init=False)
    _index_to_label: dict[int, str] = field(init=False)

    def __post_init__(self):
        self._label_to_index = {cls_name: index for index, cls_name in enumerate(self._supported_labels)}
        self._index_to_label = {index: cls_name for cls_name, index in self._label_to_index.items()}

    def get_unified_label(
        self,
        label: str
    ) -> str | None:
        """
        Returns a unified label related to the given label. Is given label is a supported one but doesn't have
        a unification label, the label itself will be returned. If given label is not a supported label, None will be returned

        Args:
            label (str): label you want to unify

        Returns:
            str | None: unified label if given label is supported else None
        """
        unified_label: str = self._unification_mapping.get(label, label)

        return unified_label if unified_label in self._supported_labels else None

    def get_index_by_label(
        self,
        label: str,
    ) -> int | None:
        """
        Label to index mapping
        """
        unified_label: str | None = self.get_unified_label(label)

        if unified_label is not None:
            return self._label_to_index.get(
                unified_label,
                None,
            )

        return None

    def get_label_by_index(
        self,
        index: int,
    ) -> str | None:
        """
        Index to label mapping
        """
        return self._index_to_label.get(
            index,
            None,
        )
    
    def get_english_label_by_index(
        self,
        index: int,
    ) -> str | None:
        """
        Index to english label mapping
        """
        return self._russian_to_english_mapping.get(
            self._index_to_label.get(
                index,
                None,
            ),
            None,
        )
    
    def get_number_of_classes(
        self,
    ) -> int:
        """
        This function is used to get proper number of supported classes

        Returns:
            int: number of supported classes
        """
        return len(self._supported_labels)


def create_label_encoder(
    path_to_unification_mapping_json: str | Path,
    path_to_supported_classes_json: str | Path,
    path_to_russian_to_english_mapping_json: str | Path,
) -> LabelEncoder:
    """
    A wrapper that used to create a LabelEncoder instance from the set of json files

    Args:
        path_to_unification_mapping_json (str | Path): path to the json file with labels unification
        path_to_supported_classes_json (str | Path): path to the json file with a list of supported classes
        path_to_russian_to_english_mapping_json (str | Path): path to the json file with proper russian to english mapping

    Returns:
        LabelEncoder: an instance of LabelEncoder
    """
    with (
        open(
            path_to_unification_mapping_json,
            'r',
            encoding='utf-8',
        ) as unification_mapping,
        open(
            path_to_supported_classes_json,
            'r',
            encoding='utf-8',
        ) as supported_classes,
        open(
            path_to_russian_to_english_mapping_json,
            'r',
            encoding='utf-8',
        ) as russian_to_english_mapping,
    ):
        return LabelEncoder(
            json.load(unification_mapping),
            json.load(supported_classes),
            json.load(russian_to_english_mapping),
        )
