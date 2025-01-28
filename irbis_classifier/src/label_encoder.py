"""This is a module for easy manipulating with labels"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelEncoder:
    _unification_mapping: dict[str, str]
    _supported_labels: tuple[str]
    _label_to_index: dict[str, int] = field(init=False)
    _index_to_label: dict[int, str] = field(init=False)

    def __post_init__(self):
        self._label_to_index = {cls_name: index for index, cls_name in enumerate(self._supported_labels)}
        self._index_to_label = {index: cls_name for cls_name, index in self._label_to_index.items()}

    def get_unified_label(
        self,
        label: str
    ) -> str | None:
        unified_label: str = self._unification_mapping.get(label, label)

        return unified_label if unified_label in self._supported_labels else None

    def get_index_by_label(
        self,
        label: str,
    ) -> int | None:
        unified_label: str | None = self.get_unified_label(label)

        if unified_label is not None:
            return self._label_to_index.get(unified_label, None)

        return None

    def get_label_by_index(
        self,
        index: int,
    ) -> str | None:
        return self._index_to_label.get(index, None)


def create_label_encoder(
    path_to_unification_mapping_json: str | Path,
    path_to_supported_classes_json: str | Path,
) -> LabelEncoder:
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
    ):
        return LabelEncoder(json.load(unification_mapping), json.load(supported_classes))
