from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset



class AnimalDataset(Dataset):
    def __init__(
        self,
        path_to_split_file: str | Path,
        transforms: A.Compose,
    ):
        super().__init__()

        self._df: pd.DataFrame = pd.read_csv(path_to_split_file)
        self._transforms = transforms

    def __len__(self) -> int:
        return self._df.shape[0]
        
    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor, int]:
        current_serie: pd.Series = self._df.loc[index]
        image_np: np.ndarray = np.asarray(Image.open(current_serie.path))

        height, width, _ = image_np.shape

        x_lower_left: int = round((current_serie.x_center - current_serie.width / 2) * width)
        x_lower_left = max(x_lower_left, 0)
        x_upper_right: int = round((current_serie.x_center + current_serie.width / 2) * width)

        y_lower_left: int = round((current_serie.y_center - current_serie.height / 2) * height)
        y_lower_left = max(y_lower_left, 0)
        y_upper_right: int = round((current_serie.y_center + current_serie.height / 2) * height)

        crop = image_np[y_lower_left:y_upper_right, x_lower_left:x_upper_right, :]
        crop = self._transforms(image=crop)['image']
        crop = crop / 255

        return crop, current_serie.class_id
