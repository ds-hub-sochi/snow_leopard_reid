from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from diffusers import FluxPipeline
from loguru import logger
from PIL import Image

from irbis_classifier.src.synthetic_data.confings import GenerationConfig


@click.command()
@click.option(
    '--path_to_config',
    type=click.Path(exists=True),
    help='path to the json config file with generation details',
)
@click.option(
    '--dump_dir',
    type=click.Path(),
    help='directory where generated images will be stored',
)
@click.option(
    '--num_images_per_prompt',
    type=int,
    default=50,
    help="number of images that model will parallel generate. Depend on your GPU's capacity"
)
def main(
    path_to_config: str | Path,
    dump_dir: str | Path,  
    num_images_per_prompt: int = 50,  
):
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: GenerationConfig = GenerationConfig(json.load(json_file))

    dump_dir: Path = Path(dump_dir).resolve()

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16,
    ).to(device)

    for specied_config in config:
        logger.info(f'generating examples for the {specied_config.russian_label}')

        current_dump_dir: Path = dump_dir / specied_config.russian_label
        current_dump_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

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
                    image.save(current_dump_dir / f'{number_of_images}.jpg')
                    number_of_images -= 1

        logger.success(f'ended with {specied_config.russian_label}')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter