from pathlib import Path

import click
import torch
from loguru import logger

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.models.factory import Factory
from irbis_classifier.src.utils import save_model_as_traced


@click.command()
@click.option('--model_name', type=str, help ="model name")
@click.option('--path_to_weight', type=click.Path(exists=True), help ="path to model's weights")
@click.option(
    '--path_to_unification_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with unification mapping',
)
@click.option(
    '--path_to_supported_labels_json',
    type=click.Path(exists=True),
    help='The path to the json file with the list of supported labels',
)
@click.option(
    '--path_to_russian_to_english_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with the russian to english mapping',
)
@click.option(
    '--path_to_traced_model_checkpoint',
    type=click.Path(),
    help='The path to the json file with the russian to english mapping',
)
def run_saving(
    model_name: str,
    path_to_weight: str | Path,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
    path_to_traced_model_checkpoint: Path | str,
) -> None:
    path_to_unification_mapping_json = Path(path_to_unification_mapping_json).resolve()
    path_to_supported_labels_json = Path(path_to_supported_labels_json).resolve()
    path_to_russian_to_english_mapping_json = Path(path_to_russian_to_english_mapping_json).resolve()

    path_to_traced_model_checkpoint = Path(path_to_traced_model_checkpoint).resolve()

    checkpoint_dir: Path = Path('/'.join(str(path_to_traced_model_checkpoint).split('/')[:-1]))
    checkpoint_dir.mkdir(
        exist_ok=True,
        parents=True,
    )


    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    model: torch.nn.Module = Factory.build_model(
        model_name,
        label_encoder.get_number_of_classes(),
    )
    model.load_state_dict(
        torch.load(
            path_to_weight,
            map_location='cpu',
        )
    )
    model.eval()

    sample_input: torch.Tensor = torch.ones((1, 3, 224, 224))

    output_before: torch.Tensor = model(sample_input)

    save_model_as_traced(
        model,
        sample_input,
        path_to_traced_model_checkpoint,
    )

    traced_model: torch.jit.ScriptModule = torch.jit.load(path_to_traced_model_checkpoint)

    output_after: torch.Tensor = traced_model(sample_input)

    if torch.allclose(output_before, output_after):
        logger.success('traced model passed the correctness test')
    else:
        logger.error("traced model didn't pass the cirrectness test")


if __name__ == "__main__":
    run_saving()  # pylint: disable=no-value-for-parameter
