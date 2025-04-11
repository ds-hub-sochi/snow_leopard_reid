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
def main(
    path_to_config: str | Path,
    dump_dir: str | Path,    
):
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: GenerationConfig = GenerationConfig(json.load(json_file))

    dump_dir: Path = Path(dump_dir).resolve()
    dump_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    num_images_per_prompt: int = 40

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16,
    ).to(device)

    for i in range(len(config)):
        logger.info(f'generating examples for the {config[i].label}')

        current_dump_dir: Path = dump_dir / config[i].label
        current_dump_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        for prompt, prompt_2, negative_prompt, negative_prompt_2 in zip(
            config[i].prompt,
            config[i].prompt_2,
            config[i].negative_prompt,
            config[i].negative_prompt_2,
        ):
            number_of_images = config[i].number_of_images

            if '{}' in prompt:
                prompt = prompt.format(config[i].label)
            if '{}' in prompt:
                prompt_2 = prompt_2.format(config[i].label)
            if '{}' in negative_prompt:
                negative_prompt = negative_prompt.format(config[i].label)
            if '{}' in negative_prompt_2:
                negative_prompt_2 = negative_prompt_2.format(config[i].label)

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
                    guidance_scale=config[i].guidance_scale,
                    num_inference_steps=30,
                    num_images_per_prompt=n_images_to_generate,
                ).images

                for image in images:
                    image.save(current_dump_dir / f'{config[i].label}_{number_of_images}.jpg')
                    number_of_images -= 1

        logger.success(f'ended with {config[i].label}')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter