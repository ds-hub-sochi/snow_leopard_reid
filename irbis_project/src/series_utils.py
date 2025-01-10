from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


DATE_COLUMN = 'date'
EXIF_DATE_KEY = 306
SEQUENCE_COLUMN = 'sequence'
INDIVIDUAL_NAME_COLUMN = 'specie'
PATHS_COLUMN = 'path'
CLASS_NAME_COLUMN = 'specie'
MAX_N_SPECIES = 'max_n_species'
COLON_STR = ':'


def get_date_from_exif(path_to_image: Path) -> str | None:
    img: Image.Image = Image.open(path_to_image)
    exif: dict[Any, Any] = dict(img.getexif())  # yes, it's fucking ugly, but getexif isn't well-typed so ...
    date: str | None = exif.get(EXIF_DATE_KEY, None)
    if date:
        date = date.replace('/', COLON_STR)  # sometimes exif date could be xx:xx:xx, sometimes xx/xx/xx

    return date


def mark_series(
    individual_df: pd.DataFrame,
    time_delta: pd.Timedelta,
) -> pd.DataFrame:
    # function to set series marks
    individual_df['date_diff'] = individual_df[DATE_COLUMN].diff()
    # check if delta is bigger than time_delta
    individual_df[SEQUENCE_COLUMN] = individual_df['date_diff'].apply(lambda date: date > time_delta)
    # set first series to 1
    # set series to 1 if it is not first and previous is 1
    individual_df[SEQUENCE_COLUMN] = individual_df[SEQUENCE_COLUMN].cumsum()

    return individual_df


def get_date_from_name(path_to_file: Path) -> str | None:
    datetime: str | None = ''
    patterns: tuple[str, str, str] = (
        r'\d{4}-\d{2}-\d{2}_\d{2}x\d{2}x\d{2}',
        r'(\d{8}) (\d{2}\.\d{2}\.\d{2})',
        r'(\d{8})_(\d{6})',
    )

    for pattern in patterns:
        match: re.Match | None = re.search(pattern, str(path_to_file))
        if match:
            datetime = process_pattern_match_by_regexp(pattern, match)
            break

    return datetime


def process_pattern_match_by_regexp(
    regexp: str,
    match: re.Match,
) -> str | None:
    datetime: str | None = None
    # extracts '2010:12:30 23:02:18' from 'PhL10-114_Suvorovka verh_FR07_20101230_230218_IMG_0015_Serga.JPG'
    if regexp == r'\d{4}-\d{2}-\d{2}_\d{2}x\d{2}x\d{2}':
        datetime = match.group()

        if datetime:
            datetime = datetime.replace('x', COLON_STR)
            datetime = datetime.replace('_', ' ')
            datetime = datetime.replace('-', COLON_STR)

    # extracts '2008:11:07 14:56:19' from 'Hernandez-Blanco20081107 14.56.19 Khromoy.JPG'
    elif regexp == r'(\d{8}) (\d{2}\.\d{2}\.\d{2})':
        date_str: str | None = match.group(1)
        time_str: str | None = match.group(2)

        if date_str and time_str:
            datetime = f'{date_str[:4]}:{date_str[4:6]}:{date_str[6:]} {time_str.replace(".", COLON_STR)}'

    # extracts '2014:05:11 20:47:30' from 'PhL14-022_Kamenka verh_10a_20140511_204730_IMG_0178_BANZAI.JPG'
    elif regexp == r'(\d{8})_(\d{6})':
        date_str = match.group(1)
        time_str = match.group(2)

        if date_str and time_str:
            datetime = f'{date_str[:4]}:{date_str[4:6]}:{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}'

    return datetime


def parse_date(path_to_file: Path) -> str | None:
    date: str | None = get_date_from_exif(path_to_file)
    if not date:
        date = get_date_from_name(path_to_file)

    return date


def add_series_info(
    df: pd.DataFrame,
    series_timedelta: str = '1 hour',
) -> pd.DataFrame:
    # find date of creation for each image
    df[DATE_COLUMN] = pd.to_datetime(
        df[PATHS_COLUMN].apply(parse_date),
        format='%Y:%m:%d %H:%M:%S',  # noqa: WPS323 the only format is strftime
        errors='coerce',
    )
    df[SEQUENCE_COLUMN] = pd.Series(dtype=int)
    # find images without exif date
    df_without_exif_dates: pd.DataFrame = df[df[DATE_COLUMN].isna()]

    classes: list[str] = sorted(df[CLASS_NAME_COLUMN].unique())
    df = df[df[DATE_COLUMN].notnull()]

    df_with_series: pd.DataFrame = find_series_per_individual(df, classes, series_timedelta)
    df_with_series.drop(columns=['date_diff'], inplace=True)

    # Each photo without exif and dates in names is considered as unique series. So, we use simple range
    # counter to mark series and add max series value per individual (specie) and + 1
    # (because range start from 0 till n-1)
    for individual in df_without_exif_dates[CLASS_NAME_COLUMN].unique():
        series_max_value: int = max(
            df_with_series.loc[df_with_series[INDIVIDUAL_NAME_COLUMN] == individual, SEQUENCE_COLUMN].values,
        )

        df_without_exif_dates.loc[df_without_exif_dates[INDIVIDUAL_NAME_COLUMN] == individual, SEQUENCE_COLUMN] = [
            idx + series_max_value + 1
            for idx in range(
                len(df_without_exif_dates[df_without_exif_dates[INDIVIDUAL_NAME_COLUMN] == individual]),
            )
        ]

    df_without_exif_dates[SEQUENCE_COLUMN] = df_without_exif_dates[SEQUENCE_COLUMN].astype(int).astype(str)

    df_with_series[INDIVIDUAL_NAME_COLUMN] = df_with_series[INDIVIDUAL_NAME_COLUMN].astype(str)
    df_with_series[SEQUENCE_COLUMN] = df_with_series[SEQUENCE_COLUMN].astype(str)

    df_with_series = pd.concat([df_with_series, df_without_exif_dates], axis=0, ignore_index=True)

    df_with_series[SEQUENCE_COLUMN] = (
        df_with_series[INDIVIDUAL_NAME_COLUMN] + '_' + df_with_series[SEQUENCE_COLUMN]  # noqa: WPS336
    )

    return df_with_series.sort_values(by=[INDIVIDUAL_NAME_COLUMN, PATHS_COLUMN]).reset_index(drop=True)


def find_series_per_individual(
    df_with_exif_dates: pd.DataFrame,
    individuals: list[str],
    series_timedelta: str,
) -> pd.DataFrame:
    df_with_series_list: list[pd.DataFrame] = []

    for individual in tqdm(individuals):
        individual_df: pd.DataFrame = df_with_exif_dates[df_with_exif_dates[INDIVIDUAL_NAME_COLUMN] == individual]
        individual_df = individual_df.sort_values(by=[DATE_COLUMN]).reset_index(drop=True)
        individual_df = mark_series(individual_df, pd.Timedelta(series_timedelta))
        df_with_series_list.append(individual_df)

    return pd.concat(df_with_series_list)
