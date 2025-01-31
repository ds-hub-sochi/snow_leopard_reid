import pandas as pd

from irbis_classifier.src.utils import sample_from_dataframe


def test_returned_df_shape():
    sample_size: int = 5

    df: pd.DataFrame = pd.DataFrame(
        [(1, 1,)] * (2 * sample_size),
        columns=['a', 'b'],
    )

    assert sample_from_dataframe(df, sample_size).shape[0] == sample_size


def test_same_resample_size():
    sample_size: int = 5

    df: pd.DataFrame = pd.DataFrame(
        [(1, 1,)] * sample_size,
        columns=['a', 'b'],
    )

    assert sample_from_dataframe(df, sample_size).shape[0] == sample_size
