from __future__ import annotations

import json
from pathlib import Path

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
    default=50,
    help="number of images that model will parallel generate. Depend on your GPU's capacity"
)
def main(
    path_to_config: str | Path,
    path_to_unification_mapping_json: str | Path,
    path_to_supported_labels_json: str | Path,
    path_to_russian_to_english_mapping_json: str | Path,
    num_images_per_prompt: int = 50,
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

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16,
    ).to(device)

    detection_model: pw_detection.MegaDetectorV5 = pw_detection.MegaDetectorV5(
        device=device,
        pretrained=True,
    )

    markup_list: list[tuple[str, str, int, float, float, float, float, int]] = []

    train_df: pd.DataFrame = pd.read_csv(repository_root_dir / 'data' / 'processed' / 'train.csv')

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    for specied_config in config:
        logger.info(f'generating examples for the {specied_config.russian_label}')

        current_images_dump_dir: Path = images_dump_dir / specied_config.russian_label
        current_images_dump_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        specie_index: int = label_encoder.get_index_by_label(specied_config.russian_label)

        for prompt, prompt_2, negative_prompt, negative_prompt_2 in zip(
            specied_config.prompt,
            specied_config.prompt_2,
            specied_config.negative_prompt,
            specied_config.negative_prompt_2,
        ):
            number_of_images: int = specied_config.number_of_images

            while number_of_images > 0:
                n_images_to_generate: int = min(
                    num_images_per_prompt,
                    number_of_images,
                )
                images: list[Image.Image] = pipeline(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    guidance_scale=specied_config.guidance_scale,
                    num_inference_steps=30,
                    num_images_per_prompt=n_images_to_generate,
                ).images

                for image in images:
                    detection_results: list[dict[str, float]] = detect_in_image(
                        np.asarray(image),
                        detection_model,
                    )

                    if len(detection_results) > 0:
                        for single_markup in detection_results:
                            markup_list.append(
                                (
                                    str(current_images_dump_dir / f'{number_of_images}.jpg'),
                                    specied_config.russian_label,
                                    specie_index,
                                    single_markup['x_center'],
                                    single_markup['y_center'],
                                    single_markup['width'],
                                    single_markup['height'],
                                    0,
                                )
                            )

                        image.save(current_images_dump_dir / f'{number_of_images}.jpg')

                    number_of_images -= 1

        logger.success(f'ended with {specied_config.russian_label}')

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

    markup_df.to_csv(repository_root_dir / 'data' / 'processed' / 'train.csv')

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
