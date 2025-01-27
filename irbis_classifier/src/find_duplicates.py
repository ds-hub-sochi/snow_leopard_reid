"""Module for image duplicates processing.
This code is mostly created by Aleksey Tolkachev
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from imagehash import phash
from joblib import Parallel, delayed
from loguru import logger
from PIL import Image

IMAGES_KW: str = 'images'
VIDEOS_KW: str = 'videos'

VALID_IMG_EXTS: tuple[str, str, str] = ('.jpg', '.jpeg', '.png')
VALID_VIDEO_EXTS: tuple[str, str, str] = ('.mp4', '.mov', '.avi')


class DuplicateOpsProcessor:
    """Auxillary class for file processing given path to json with image and video duplicates."""
    def __init__(
        self,
        hashes_json_path: str,
        n_jobs: int = -1,
    ):
        self.n_jobs: int = n_jobs

        with open(
            hashes_json_path,
            'r',
            encoding='utf-8',
        ) as jf:
            self.hashes = json.loads(jf.read())

    def copy_files_for_manual_check(
        self,
        out_dir: Path,
    ) -> bool:
        """Given dict with hashes from class attribute copies duplicates to provided out dir for manual checking.

        Args:
            out_dir (Path): Path to output dir, where duplicates would be copied

        Returns:
            bool: flag for successful processing
        """

        counter: int = 0
        for _, image_paths in self.hashes[IMAGES_KW].items():
            if len(image_paths) > 1:
                counter += 1

                (out_dir / IMAGES_KW / f'{Path(image_paths[0]).parent.stem}_{counter}').mkdir(
                    parents=True,
                    exist_ok=True,
                )

                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._copy_file)(
                        file_path,
                        idx,
                        out_dir / IMAGES_KW / f'{Path(image_paths[0]).parent.stem}_{counter}',
                    )
                    for idx, file_path in enumerate(image_paths, 1)
                )

        for _, video_groups_by_size in self.hashes[VIDEOS_KW].items():
            for _, video_paths in video_groups_by_size.items():
                if len(video_paths) > 1:
                    counter += 1

                    (out_dir / VIDEOS_KW / f'{Path(video_paths[0]).parent.stem}_{counter}').mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._copy_file)(
                            file_path,
                            idx,
                            out_dir / VIDEOS_KW / f'{Path(video_paths[0]).parent.stem}_{counter}',
                        )
                        for idx, file_path in enumerate(video_paths, 1)
                    )

        return True

    def remove_duplicates(self) -> bool:
        """Removes duplicate media files given paths from {hash:[paths]} structure if len(path) > 1
        for images and {file_sizes: {hash:[paths]} for videos

        Returns:
            bool: flag for successful processing
        """
        for _, imgs in self.hashes[IMAGES_KW].items():
            if len(imgs) > 1:
                self._remove_files(imgs[1:])
        for _, hashes_dict in self.hashes[VIDEOS_KW].items():
            for _, videos in hashes_dict.items():
                if len(videos) > 1:
                    self._remove_files(videos[1:])

        return True

    def _remove_files(
        self,
        file_paths: list[str],
    ) -> bool:
        """Removes files given a list of string paths

        Args:
            file_paths (list): List of string file paths to be removed

        Returns:
            bool: flag for successful processing
        """
        for file_path in file_paths:
            if Path(file_path).exists():
                Path(file_path).unlink()

        return True

    def _copy_file(
        self,
        file_path: str | Path,
        counter: int,
        out_dir: Path,
    ) -> bool:
        """ Copy the file to the output directory with the modified name

        Args:
            file_path (str): Path for file which would be copied
            counter (int): int meta for unique name generation
            out_dir (Path): Path to output dir, where duplicates would be copied

        Returns:
            bool: flag for successful processing
        """
        file_path = Path(file_path)
        out_path: Path = out_dir / f'{file_path.stem}_{counter}{file_path.suffix}'
        logger.debug(out_path)
        shutil.copy2(file_path, out_path)

        return True


class DuplicateFinder:
    """Class for image and video duplicates searching based on perceptual hashing algo"""
    def __init__(
        self,
        src_dir: str,
        hash_size: int = 32,
        highfreq_factor: int = 8,
        n_jobs: int = -1,
    ):
        self.src_dir: Path = Path(src_dir)
        self.image_hashes: defaultdict[str, list] = defaultdict(list)
        self.video_hashes: defaultdict[str, defaultdict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._video_sizes: defaultdict[str, list] = defaultdict(list)
        self.hash_size: int = hash_size
        self.hf_factor: int = highfreq_factor
        self.n_jobs: int = n_jobs

    def hash_image(
        self,
        image_path: Path,
    ) -> str:
        """Hash the image using perceptual hash method.

         Args:
            image_path (Path): path for image, for which hash would be computed

        """
        img = Image.open(image_path)

        return str(
            phash(
                img,
                hash_size=self.hash_size,
                highfreq_factor=self.hf_factor,
            )
        )

    def hash_video(
        self,
        video_path: Path,
        fps_factor: int = 1,
    ) -> str:
        """Create a hash for the video based on its frames.

        Args:
            video_path (Path): path for video, for which hash would be computed
            fps_factor (int, optional): Seconds multiplier. If >= 1, then hash would be calculated
            each fps_factor second. Defaults to 1.

        """

        logger.info(
            f"processing {'/'.join(Path(video_path).parts[-2:])}",
        )

        cap = cv2.VideoCapture(str(video_path))  # pylint: disable=no-member
        hash_list: list[str] = []
        frame_counter: int = 0
        broken_frames: int = 0
        while cap.isOpened() and frame_counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # pylint: disable=no-member
            success, frame = cap.read()
            if success:
                if frame_counter % (int(cap.get(cv2.CAP_PROP_FPS)) * fps_factor) == 0:  # pylint: disable=no-member
                    hash_list.append(
                        str(
                            phash(
                                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),  # pylint: disable=no-member
                                hash_size=self.hash_size,
                                highfreq_factor=self.hf_factor,
                            ),
                        ),
                    )
            else:
                broken_frames += 1
            frame_counter += 1
        if broken_frames:
            logger.warning(
                '{0} frames in file. {1} out if {2} frames could not be read in {3}'.format(  # pylint: disable=consider-using-f-string  # noqa: E501
                    frame_counter,
                    broken_frames,
                    frame_counter // (int(cap.get(cv2.CAP_PROP_FPS)) * fps_factor),
                    '/'.join(Path(video_path).parts[-2:]),
                ),
            )
        cap.release()

        return ''.join(hash_list)

    def find_duplicates(self) -> tuple[dict, dict]:
        """Find duplicates in the directory."""

        if self.src_dir.is_dir():
            file_paths: list[Path] = list(self.src_dir.rglob('*'))
        else:
            file_paths = list(pd.read_csv(self.src_dir).path)

        file_props = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_file)(Path(file_path)) for file_path in file_paths
        )

        for prop in file_props:
            if prop is None:
                continue
            if prop[0] == 'image':
                _, file_hash, file_path = prop
                self.image_hashes[file_hash].append(file_path)
            elif prop[0] == 'video':
                _, file_size, file_path = prop
                self._video_sizes[file_size].append(file_path)

        self._calculate_video_hashes()

        return self.image_hashes, self.video_hashes

    def _process_file(
        self,
        file_path: Path,
    ) -> tuple[str, str, str] | tuple[str, int, str] | None:
        """Process a single file to generate its hash in case of image or group file by its size in case of video."""
        try:
            if file_path.suffix.lower() in VALID_IMG_EXTS:
                file_hash = self.hash_image(file_path)

                return ('image', file_hash, str(file_path))

            if file_path.suffix.lower() in VALID_VIDEO_EXTS:
                file_size = file_path.stat().st_size

                return ('video', file_size, str(file_path))
        except (ValueError, TypeError) as ex:
            logger.error(f'error processing file {file_path}: {ex}')

        return None

    def _process_video(
        self,
        file_path: Path,
        file_size: int,
    ) -> tuple[int, str, str]:
        """Process a single video from the given file_size group to generate its hash.

        Args:
            file_path (Path): Path to video
            file_size (int): Value of the video size in bytes from Path.st_size

        Returns:
            tuple[int, str, str]: meta tuple of size, hash and path for futher data packing
        """
        video_hash: str = self.hash_video(file_path)

        return (file_size, video_hash, str(file_path))

    def _calculate_video_hashes(self) -> bool:
        """Calculate hashes for video files that have the same size."""

        video_hashes = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_video)(file_path, file_size)
            for file_size, file_paths in self._video_sizes.items()
            if len(file_paths) > 1
            for file_path in file_paths
        )

        for file_size, video_hash, file_path in video_hashes:
            self.video_hashes[file_size][video_hash].append(file_path)

        return True


def export_dict2json(
    dict_struct: dict[Any, Any],
    json_path: str,
) -> bool:
    """General function to export dict structure to .json file

    Args:
        dict_struct (dict): dict with proper data
        json_path (str): file path where .json would be saved

    Returns:
        bool: flag for successful processing

    """
    with Path(json_path).open(
        mode='w',
        encoding='utf-8',
    ) as jf:
        json.dump(
            dict_struct,
            jf,
            indent=4,
            ensure_ascii=False,
        )

    return True


if __name__ == '__main__':
    directory = 'hash_test'
    finder: DuplicateFinder = DuplicateFinder(directory)
    img_hashes, video_hashes = finder.find_duplicates()

    export_dict2json(
        {
            IMAGES_KW: img_hashes,
            VIDEOS_KW: video_hashes,
        },
        './hashes.json',
    )

    ops_processor = DuplicateOpsProcessor('./hashes.json')
    # ops_processor.copy_files_for_manual_check(Path('./duplicates'))
    ops_processor.remove_duplicates()
