from typing import Any, Dict, List
from pathlib import Path

from fabric.config import os
from higgsfield.internal.experiment.decorator import _experiments
from higgsfield.internal.experiment.params import parse_kwargs_to_params
from higgsfield.internal.experiment.builder import _source_experiments


class Launch:
    experiment_name: str
    run_name: str
    max_repeats: int
    kwargs: Dict[str, str]
    prepared: Any

    def __init__(
        self,
        wd: Path,
        project_name: str,
        experiment_name: str,
        run_name: str,
        max_repeats: int,
        rest: List[str],
    ):
        if not experiment_name or experiment_name == "":
            raise ValueError("Experiment name cannot be empty")

        if not project_name or project_name == "":
            raise ValueError("Project name cannot be empty")

        if not run_name or run_name == "":
            raise ValueError("Run name cannot be empty")

        if not max_repeats or max_repeats < -1:
            raise ValueError("Max repeats cannot be none or less than -1")

        self.wd = wd
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.run_name = run_name
        self.max_repeats = max_repeats
        self.kwargs = self._parse(rest)
        self._find_route()
        self.eval_params()
        self.apply_train()

    def _parse(self, rest: List[str]):
        kwargs = {}
        for arg in rest:
            if "=" not in arg:
                continue
            key, value = arg.split("=")
            kwargs[key] = value
        return kwargs

    def _find_route(self):
        _source_experiments(self.wd / "src")

        if self.experiment_name not in _experiments:
            raise ValueError(f"Experiment {self.experiment_name} not found")

        experiment = _experiments[self.experiment_name]
        self.experiment = experiment

    def eval_params(self):
        params = parse_kwargs_to_params(self.experiment.params, self.kwargs)
        setattr(params, "experiment_name", self.experiment_name)
        setattr(params, "project_name", self.project_name)
        setattr(params, "run_name", self.run_name)
        setattr(params, "rank", int(os.environ.get("RANK", 0)))
        setattr(params, "world_size", int(os.environ.get("WORLD_SIZE", 1)))
        setattr(params, "local_rank", int(os.environ.get("LOCAL_RANK", 0)))

        self.prepared = params

    def apply_train(self):
        self.experiment.train(self.prepared)
