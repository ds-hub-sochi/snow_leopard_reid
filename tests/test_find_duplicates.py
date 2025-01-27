from pathlib import Path

from irbis_classifier.src.find_duplicates import DuplicateFinder


def test_path_to_dir_as_input():
    test_dir: Path = Path(__file__).parent.resolve()

    path: str = str(test_dir / 'assets' / 'test_find_duplicates')

    finder: DuplicateFinder = DuplicateFinder(path)

    img_hashes, video_hashes = finder.find_duplicates()

    assert len(video_hashes) == 0
    assert len(img_hashes) == 3

    answers: tuple[int, ...] = (3, 2, 1)

    for hash_, answer in zip(img_hashes, answers):
        assert len(img_hashes[hash_]) == answer


def test_path_to_file_as_input():
    test_dir: Path = Path(__file__).parent.resolve()

    path: str = str(test_dir / 'assets' / 'test_find_duplicates' / 'csv_input.csv')

    finder: DuplicateFinder = DuplicateFinder(path)

    img_hashes, video_hashes = finder.find_duplicates()

    assert len(video_hashes) == 0
    assert len(img_hashes) == 3

    answers: tuple[int, ...] = (3, 2, 1)

    for hash_, answer in zip(img_hashes, answers):
        assert len(img_hashes[hash_]) == answer
