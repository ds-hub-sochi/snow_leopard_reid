from os import remove
from pathlib import Path

import pandas as pd

from irbis_classifier.src.find_duplicates import DuplicateFinder, DuplicateOpsProcessor, \
    export_dict2json, VIDEOS_KW, IMAGES_KW


test_dir: Path = Path(__file__).parent.resolve()


def test_path_to_dir_as_input():
    path: str = str(test_dir / 'assets' / 'test_find_duplicates')

    finder: DuplicateFinder = DuplicateFinder(path)

    img_hashes, video_hashes = finder.find_duplicates()

    assert len(video_hashes) == 0
    assert len(img_hashes) == 3

    answers: tuple[int, ...] = (3, 2, 1)

    for hash_, answer in zip(img_hashes, answers):
        assert len(img_hashes[hash_]) == answer


def test_path_to_file_as_input():
    path: str = str(test_dir / 'assets' / 'test_find_duplicates' / 'csv_input.csv')

    finder: DuplicateFinder = DuplicateFinder(path)

    img_hashes, video_hashes = finder.find_duplicates()

    assert len(video_hashes) == 0
    assert len(img_hashes) == 3

    answers: tuple[int, ...] = (3, 2, 1)

    for hash_, answer in zip(img_hashes, answers):
        assert len(img_hashes[hash_]) == answer


def test_removing_from_markup_file():
    base_path: Path = test_dir / 'assets' / 'test_find_duplicates'

    df: pd.DataFrame = pd.read_csv(str(base_path / 'csv_input.csv'))
    for i in range(df.shape[0]):
        df.loc[i, 'path'] = str(Path(df.loc[i, 'path']).resolve())
    df.to_csv(str(base_path / 'csv_input_resolved_path.csv'), index=None)

    finder: DuplicateFinder = DuplicateFinder(base_path / 'csv_input_resolved_path.csv')

    img_hashes, video_hashes = finder.find_duplicates()

    export_dict2json(
        {
            IMAGES_KW: img_hashes,
            VIDEOS_KW: video_hashes,
        },
        test_dir / 'hashes.json',
    )

    ops_processor: DuplicateOpsProcessor = DuplicateOpsProcessor(test_dir / 'hashes.json')
    df_filtered: pd.DataFrame = ops_processor.remove_duplicates_from_markup_file(
        base_path / 'csv_input_resolved_path.csv',
    )

    remove(test_dir / 'hashes.json')
    remove(base_path / 'csv_input_resolved_path.csv')

    assert df_filtered.shape[0] == 3
