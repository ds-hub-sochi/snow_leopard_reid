from __future__ import annotations

import glob
import json
import zipfile
from os import walk, rename
from pathlib import Path
from shutil import copytree, rmtree

import click
import cv2
from loguru import logger


N_IMAGES_PER_FRAME: int = 2

ORIGIANAL_NAMING_TO_RUSSIAN: dict[str, str] = {
    'alces': 'Лось',
    'bison': 'Зубр',
    'canis lupus': 'Волк',
    'capreolus': 'Косуля',
    'castor': 'Бобр',
    'cnippon': 'Пятнистый олень',
    'lepus': 'Заяц',
    'lutra': 'Выдра',
    'lynx': 'Рысь',
    'martes': 'Харза',  # 'Куница'
    'meles': 'Барсук',
    'mm': 'Харза',  # 'Куница'
    'mustela': 'Ласка(Горностай)',
    'neovison': 'Норка',
    'nyctereutes': 'Енотовидная собака',
    'putorius': 'Хорь',
    'mputorius': 'Хорь',
    'sus': 'Кабан',
    'vulpes': 'Лиса',
    'ursus': 'Медведь',
}  # Other animals are ignored cause their unknown type will be a probliem to distinguish them in the future


@click.command()
@click.option(
    '--path_to_data',
    type=click.Path(exists=True),
    help='The path to the data directory',
)
def add_data(
    path_to_data: str | Path,
) -> None:  # pylint: disable=too-many-locals
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    next_stage_index: int = len(glob(str(repository_root_dir / 'data' / 'raw' / 'full_images' / '*'))) + 1
    save_dir: Path = repository_root_dir / 'data' / 'raw'
    for subdir in ('full_images', 'detection_labels'):
        (save_dir / subdir / f'stage_{next_stage_index}').mkdir(
            exist_ok=True,
            parents=True,
        )

    temp_dir: Path = save_dir / 'temp'

    path_to_data = Path(path_to_data).resolve()

    content_dir: Path = path_to_data / 'content'
    markup_dir: Path = path_to_data / 'markup'

    copytree(content_dir, temp_dir / 'content', dirs_exist_ok=True)
    copytree(markup_dir, temp_dir / 'markup', dirs_exist_ok=True)
    logger.info(f'content was replaced into the temprorary dir {temp_dir}')

    content_dir = temp_dir / 'content'
    markup_dir = temp_dir / 'markup'

    years: list[str] = []
    for (_, dirnames, _) in walk(markup_dir):
        years.extend(dirnames)
        break

    for year in years:
        current_markup_dir: Path = markup_dir / year
        archives: list[str] = glob.glob(str(current_markup_dir / '*.zip'))

        for archive in archives:
            filename: str = archive.split('/')[-1][:-4]

            with zipfile.ZipFile(
                archive,
                'r',
            ) as zip_ref:
                zip_ref.extractall(current_markup_dir / filename)

    # now it's important to make all animals dirs in lowercase
    for year in years:
        current_markup_dir = markup_dir / year
        current_content_dir = content_dir / year / 'VIDEO'

        for dir in {current_markup_dir, current_content_dir}:
            labels = []
            for (_, dirnames, _) in walk(dir):
                labels.extend(dirnames)
                break

            for label in labels:
                rename(
                    str(dir / label),
                    str(dir / label.lower()),
                )

    for year in years:
        current_markup_dir = markup_dir / year

        labels: list[str] = []
        for (_, dirnames, _) in walk(current_markup_dir):
            labels.extend(dirnames)
            break

        for label in labels:
            if not (content_dir / year / 'VIDEO' / label[:-5]).is_dir():
                logger.error(f"content for {label[:-5]} wasn't found")

        for label in labels:
            if label[:-5] in ORIGIANAL_NAMING_TO_RUSSIAN:
                russian_label: str = ORIGIANAL_NAMING_TO_RUSSIAN[label[:-5]]

                jsons: list[str] = glob.glob(str(current_markup_dir / label / '*.json'))
                label = label[:-5]
                logger.info(f'now working with {label} class')

                current_images_save_dir: Path = repository_root_dir / 'data' / 'raw' / 'full_images' / \
                    f'stage_{next_stage_index}' / russian_label
                current_images_save_dir.mkdir(
                    exist_ok=True,
                    parents=True,
                )

                current_markup_save_dir: Path = repository_root_dir / 'data' / 'raw' / 'detection_labels' / \
                    f'stage_{next_stage_index}' / russian_label
                current_markup_save_dir.mkdir(
                    exist_ok=True,
                    parents=True,
                )

                for json_path in jsons:
                    video_name: str = json_path.split('\\')[-1][:-5] + ".AVI"

                    with open(
                        json_path,
                        'r',
                        encoding='utf-8',
                    ) as f:
                        current_markup = json.load(f)

                    for key in list(current_markup['detections'].keys()):
                        current_markup['detections'][int(key)] = current_markup['detections'][key]
                        del current_markup['detections'][key]

                    current_markup['detections'] = dict(sorted(current_markup['detections'].items()))
                    keys: list[int] = list(current_markup['detections'].keys())

                    video_capture = cv2.VideoCapture(content_dir / year / 'VIDEO' / label / video_name)
                    fps: int = int(video_capture.get(cv2.CAP_PROP_FPS))

                    success, image = video_capture.read()
                    count: int = 0

                    while success and count < len(list(current_markup['detections'])):
                        current_bboxes: list[dict[int, float]] = \
                            [elem['bbox'] for elem in current_markup['detections'][keys[count]] if elem['conf'] > 0.5]
                        
                        if len(current_bboxes) > 0:
                            with open(
                                current_markup_save_dir / f'{video_name[:-4]}_{count}.txt',
                                'w',
                                encoding='utf-8',
                            ) as markup_file:

                                for current_bbox in current_bboxes:
                                    upper_left_x: float = current_bbox[0]
                                    upper_left_y: float = current_bbox[1]
                                    width: float = current_bbox[2]
                                    height: float = current_bbox[3]

                                    x_center: float = upper_left_x + width / 2
                                    y_center: float = upper_left_y + height / 2

                                    markup_file.write(f'{0} {x_center} {y_center} {width} {height}\n')

                            cv2.imwrite(
                                current_images_save_dir / f'{video_name[:-4]}_{count}.JPG',
                                image,
                            )
                        count += (fps // N_IMAGES_PER_FRAME)
                        video_capture.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            count,
                        )
                        success, image = video_capture.read()
                logger.success(f'ended with {label}')

    rmtree(str(temp_dir))
    logger.info('temprorary dir was remodev')


if __name__ == '__main__':
    add_data()  # pylint: disable=no-value-for-parameter
