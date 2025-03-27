from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from loguru import logger

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.models.factory import Factory
from irbis_classifier.src.utils import save_model_as_traced


@dataclass
class LabelEncoderParams:
    path_to_unification_mapping_json: str | Path
    path_to_supported_labels_json: str | Path
    path_to_russian_to_english_mapping_json: str | Path

    def __post_init__(self):
        self.path_to_unification_mapping_json = Path(self.path_to_unification_mapping_json).resolve()
        self.path_to_supported_labels_json = Path(self.path_to_supported_labels_json).resolve()
        self.path_to_russian_to_english_mapping_json = Path(self.path_to_russian_to_english_mapping_json).resolve()


@dataclass
class SavingConfig:
    model_name: str
    path_to_weight: str | Path
    path_to_traced_model_checkpoint: str | Path
    input_size: int
    label_encoder_params: LabelEncoderParams

    def __post_init__(self):
        self.path_to_weight = Path(self.path_to_weight).resolve()

        self.path_to_traced_model_checkpoint = Path(self.path_to_traced_model_checkpoint).resolve()

        self.label_encoder_params = LabelEncoderParams(**self.label_encoder_params)  # pylint: disable=not-a-mapping


@click.command()
@click.option(
    '--path_to_config',
    type=click.Path(exists=True),
    help='path to the json-config file',
)
def run_saving(
    path_to_config: str,
) -> None:
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: SavingConfig = SavingConfig(**json.load(json_file))

    checkpoint_dir: Path = Path(config.path_to_traced_model_checkpoint).parent
    if not checkpoint_dir.exists():
        logger.warning("directory you've provided as a checkpoint doesn't exists; it was manually created")
        checkpoint_dir.mkdir(
            parents=True,
        )

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
        path_to_supported_classes_json=config.label_encoder_params.path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
    )

    model: torch.nn.Module = Factory.build_model(
        config.model_name,
        label_encoder.get_number_of_classes(),
    )
    model.load_state_dict(
        torch.load(
            config.path_to_weight,
            map_location='cpu',
        )
    )
    model.eval()

    sample_input: torch.Tensor = torch.ones((1, 3, config.input_size, config.input_size))

    output_before: torch.Tensor = model(sample_input)

    save_model_as_traced(
        model,
        sample_input,
        config.path_to_traced_model_checkpoint,
    )

    traced_model: torch.jit.ScriptModule = torch.jit.load(config.path_to_traced_model_checkpoint)

    output_after: torch.Tensor = traced_model(sample_input)

    if torch.allclose(output_before, output_after):
        logger.success('traced model passed the correctness test')
    else:
        logger.error("traced model didn't pass the correctness test")


if __name__ == "__main__":
    run_saving()  # pylint: disable=no-value-for-parameter
