from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from comet_ml import CometExperiment, start


@dataclass
class ExperimentConfig:
    api_key: str
    project_name: str
    workspace: str


def setup_experimet(
    path_to_json_config: str | Path,
    run_name: str,
) -> CometExperiment:
    """
    A wrapper that creates a commet experiment

    Args:
        path_to_json_config (str | Path): path to json config with Comet specific private information
        run_name (str): name of the current Comet run

    Returns:
        CometExperiment: an instance of a Comet experiment
    """
    with open(
        path_to_json_config,
        'r',
        encoding='utf-8',
    ) as json_config:
        experiment_config: ExperimentConfig = ExperimentConfig(
            **json.load(json_config)
        )

    expetiment: CometExperiment = start(
        api_key=experiment_config.api_key,
        project_name=experiment_config.project_name,
        workspace=experiment_config.workspace,
    )
    expetiment.set_name(run_name)

    return expetiment
