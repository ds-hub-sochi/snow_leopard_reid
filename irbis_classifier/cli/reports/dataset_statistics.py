from __future__ import annotations

from pathlib import Path

from irbis_classifier.src.plots import (
    create_pie_plots_over_split,
    create_classes_difference_over_split,
    create_bar_plot_over_stages,
)


def collect_figures():
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    create_pie_plots_over_split(
        repository_root_dir / 'data' / 'processed',
        False,
        True,
        repository_root_dir / 'reports' / 'figures' / 'splits_classes_distribution',
    )

    create_classes_difference_over_split(
        repository_root_dir / 'data' / 'processed',
        False,
        True,
        repository_root_dir / 'reports' / 'figures',
    )

    create_bar_plot_over_stages(
        repository_root_dir / 'data' / 'processed',
        False,
        True,
        repository_root_dir / 'reports' / 'figures',
    )


if __name__ == '__main__':
    collect_figures()
