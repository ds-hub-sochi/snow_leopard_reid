from pathlib import Path


def filter_non_images(image_paths: list[Path]) -> list[Path]:
    return [
        image_path for image_path in image_paths
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}
    ]


def fix_rus_i_naming(filename: str) -> str:
    # Fix inconsistency with 'й' symbol.
    # First one is quite normal in Ubuntu/Mac and presents in .json markup
    # The second one is different and presents in original filenames
    return filename.replace('й', 'й')
