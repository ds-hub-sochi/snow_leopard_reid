from pathlib import Path

import pytest
from transformers import AutoModel, AutoTokenizer
import torch

from irbis_classifier.src.series_utils import get_date_and_time_using_ocr


device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
)

model: AutoModel = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device,
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,  # type: ignore
).eval().to(device)


test_dir: Path = Path(__file__).parent.resolve()

test_args: tuple[tuple[Path, str, str], ...] = (
    (test_dir / 'assets' / 'test_ocr' / 'AmurLeopard_31.jpg', '2015:04:12', '04:51:54'),
    (test_dir / 'assets' / 'test_ocr' / 'AmurTiger_899.jpg', '2020:12:10', '05:48:28'),
    (test_dir / 'assets' / 'test_ocr' / 'nightBlackBearbunight1670.jpg', '2020:09:24', '00:24:28'),
    (test_dir / 'assets' / 'test_ocr' / 'RaccoonDog192.jpg', '2016:03:16', '19:39:47'),
    (test_dir / 'assets' / 'test_ocr' / 'RacoonDog_00211500.jpg', '2015:10:04', '19:47:00'),
    (test_dir / 'assets' / 'test_ocr' / 'RedFox_1 (508).jpg', '2015:02:05', '01:44:02'),
    (test_dir / 'assets' / 'test_ocr' / 'RoeDeer(582).jpg', '2016:03:31', '21:35:24'),
    (test_dir / 'assets' / 'test_ocr' / 'Sable98.jpg', '2016:10:30', '18:52:50'),
    (test_dir / 'assets' / 'test_ocr' / 'SablebunightSablebunight1173.jpg', '2020:03:27', '01:06:19'),
    (test_dir / 'assets' / 'test_ocr' / 'SikaDeer_2224.jpg', '2014:01:11', '05:02:32'),
    (test_dir / 'assets' / 'test_ocr' / 'SikaDeer410.jpg', '2016:04:30', '03:55:55'),
    (test_dir / 'assets' / 'test_ocr' / 'Weasel_1079.jpg', '2020:12:09', '21:21:08'),
    (test_dir / 'assets' / 'test_ocr' / 'Weasel_1266.jpg', '2020:02:23', '21:20:39'),
    (test_dir / 'assets' / 'test_ocr' / 'Wildboar_1816.jpg', '2016:09:29', '20:00:11'),
    (test_dir / 'assets' / 'test_ocr' / 'Y.T.Marten515.jpg', '2020:12:24', '06:35:49'),
    (test_dir / 'assets' / 'test_ocr' / 'AmurLeopard_8.jpg', '2015:11:20', '11:09:31'),
    (test_dir / 'assets' / 'test_ocr' / 'Badger__177.jpg', '2020:11:19', '11:58:13'),
    (test_dir / 'assets' / 'test_ocr' / 'BlackBear_102.jpg', '2015:10:07', '17:47:33'),
    (test_dir / 'assets' / 'test_ocr' / 'dayBlackBearbunight1055.jpg', '2020:04:29', '12:38:52'),
    (test_dir / 'assets' / 'test_ocr' / 'leopard_2426.jpg', '2016:08:24', '07:48:50'),
    (test_dir / 'assets' / 'test_ocr' / 'LeopardCat_4046.jpg', '2016:10:07', '14:43:09'),
    (test_dir / 'assets' / 'test_ocr' / 'MuskDeer_1595.jpg', '2017:01:29', '15:53:15'),
    # (test_dir / 'assets' / 'test_ocr' / 'nightBlackBearbunight726.jpg', '2020:10:03', '05:36:45'),
    (test_dir / 'assets' / 'test_ocr' / 'RaccoonDog254.jpg', '2016:03:31', '11:24:02'),
    (test_dir / 'assets' / 'test_ocr' / 'RacoonDog_114 (2).jpg', '2015:10:09', '06:03:04'),
    (test_dir / 'assets' / 'test_ocr' / 'RedFox(212).jpg', '2015:11:29', '11:30:21'),
    (test_dir / 'assets' / 'test_ocr' / 'RoeDeer(729).jpg', '2016:03:24', '16:19:55'),
    (test_dir / 'assets' / 'test_ocr' / 'SablebunightSablebunight976.jpg', '2020:09:29', '07:48:04'),
    (test_dir / 'assets' / 'test_ocr' / 'SikaDeer_2196.jpg', '2014:10:29', '06:01:58'),
    (test_dir / 'assets' / 'test_ocr' / 'Weasel_418.jpg', '2015:10:08', '18:20:15'),
    (test_dir / 'assets' / 'test_ocr' / 'WildBoar799.jpg', '2016:05:05', '10:38:17'),
    (test_dir / 'assets' / 'test_ocr' / 'Y.T.Marten83.jpg', '2016:01:01', '14:51:28'),
)


@pytest.mark.parametrize("filepath,expected_date,expected_time", test_args)
@torch.inference_mode()
def test_ocr_correctness(
    filepath: Path,
    expected_date: str,
    expected_time: str,
):
    """
    See https://huggingface.co/stepfun-ai/GOT-OCR2_0
    """
    date: str = get_date_and_time_using_ocr(
        model,
        tokenizer,
        filepath,
    )

    date, time = date.split(' ')

    assert date == expected_date
    assert time == expected_time
