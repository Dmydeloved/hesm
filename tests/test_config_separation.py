from __future__ import annotations

from config import DEFAULT_CONFIG_PATH as HESM_CONFIG_PATH
from config import load_config
from experiments.config import DEFAULT_CONFIG_PATH as EXPERIMENT_CONFIG_PATH
from experiments.config import load_experiment_config


def test_hesm_and_experiment_configs_are_independent() -> None:
    hesm_config = load_config()
    experiment_config = load_experiment_config()

    assert HESM_CONFIG_PATH.name == "hesm.yaml"
    assert EXPERIMENT_CONFIG_PATH.name == "locomo.yaml"
    assert HESM_CONFIG_PATH != EXPERIMENT_CONFIG_PATH
    assert hesm_config["paths"]["memory_db"].startswith("memory/")
    assert "experiment" not in hesm_config
    assert "memory_methods" not in hesm_config
    assert experiment_config["output"]["root"].startswith(
        "experiments/outputs/"
    )
    assert experiment_config["memory_methods"]["hesm"]["retrieval"]
