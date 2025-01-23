from __future__ import annotations

from irbis_classifier.src.plots import (
    create_pie_plots_over_split,
    create_classes_difference_over_split,
)


def collect_figures():
    create_pie_plots_over_split(False, True)
    create_classes_difference_over_split(False, True)


if __name__ == '__main__':
    collect_figures()
