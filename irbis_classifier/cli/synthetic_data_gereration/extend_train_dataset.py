from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from time import gmtime, strftime

import click
import numpy as np
import pandas as pd
import torch
from diffusers import FluxPipeline
from loguru import logger
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.synthetic_data.confings import GenerationConfig
from irbis_classifier.src.utils import detect_in_image


@click.command()
@click.option(
    '--path_to_config',
    type=click.Path(exists=True),
    help='path to the json config file with generation details',
)
@click.option(
    '--path_to_unification_mapping_json',
    type=click.Path(exists=True),
)
@click.option(
    '--path_to_supported_labels_json',
    type=click.Path(exists=True),
)
@click.option(
    '--path_to_russian_to_english_mapping_json',
    type=click.Path(exists=True),
)
@click.option(
    '--num_images_per_prompt',
    type=int,
    default=20,
    help="number of images that model will parallel generate. Depend on your GPU's capacity"
)
def main(  # pylint: disable=too-many-locals
    path_to_config: str | Path,
    path_to_unification_mapping_json: str | Path,
    path_to_supported_labels_json: str | Path,
    path_to_russian_to_english_mapping_json: str | Path,
    num_images_per_prompt: int = 20,
):
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: GenerationConfig = GenerationConfig(json.load(json_file))

    repository_root_dir: Path  = Path(__file__).parent.parent.parent.parent.resolve()

    images_dump_dir: Path = repository_root_dir / 'data' / 'raw' / 'full_images' / 'stage_0'
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    detection_model: pw_detection.MegaDetectorV5 = pw_detection.MegaDetectorV5(
        device=device,
        pretrained=True,
    ).eval()

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16,
    ).to(device)

    markup_list: list[tuple[str, str, int, float, float, float, float, int]] = []

    train_df: pd.DataFrame = pd.read_csv(repository_root_dir / 'data' / 'processed' / 'train.csv')

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    for species_config in config:  # pylint: diable=too-many-nested-blocks
        logger.info(f'generating examples for the {species_config.russian_label}')

        current_images_dump_dir: Path = images_dump_dir / species_config.russian_label
        current_images_dump_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        specie_index: int = label_encoder.get_index_by_label(species_config.russian_label)

        for i in range(len(species_config.prompt)):
            args_dict = asdict(species_config)

            del args_dict['russian_label']
            del args_dict['number_of_images']

            for prompt_name in ('prompt', 'prompt_2', 'negative_prompt', 'negative_prompt_2'):
                args_dict[prompt_name] = args_dict[prompt_name][i]
            
            for prompt_name in ('prompt', 'prompt_2', 'negative_prompt', 'negative_prompt_2'):
                if args_dict[prompt_name] == '':
                    del args_dict[prompt_name]

            number_of_images: int = species_config.number_of_images

            while number_of_images > 0:
                n_images_to_generate: int = min(
                    num_images_per_prompt,
                    number_of_images,
                )

                args_dict['num_images_per_prompt'] = n_images_to_generate

                images: list[Image.Image] = pipeline(
                    **args_dict,
                ).images

                for image in images:
                    detection_results: list[dict[str, float]] = detect_in_image(
                        np.asarray(image),
                        detection_model,
                    )

                    if len(detection_results) > 0:
                        current_date: str = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

                        save_path: Path = current_images_dump_dir / f'{current_date}_{i}_{number_of_images}.jpg'
                        for single_markup in detection_results:
                            markup_list.append(
                                (
                                    str(save_path),
                                    species_config.russian_label,
                                    specie_index,
                                    single_markup['x_center'],
                                    single_markup['y_center'],
                                    single_markup['width'],
                                    single_markup['height'],
                                    0,
                                )
                            )

                        image.save(save_path)

                    number_of_images -= 1

                torch.cuda.empty_cache()
                gc.collect()

        logger.success(f'ended with {species_config.russian_label}')

    markup_df: pd.DataFrame = pd.DataFrame(
        markup_list,
        columns=train_df.columns,
    )

    markup_df = pd.concat(
        [
            train_df,
            markup_df,
        ],
        axis=0,
    ).reset_index(drop=True)

    markup_df.to_csv(
        repository_root_dir / 'data' / 'processed' / 'train.csv',
        index=False,    
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
