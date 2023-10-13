import pathlib


class _ProjectCachePath:
    project_name: str
    path: pathlib.Path
    metadata_file: str
    _init_path: pathlib.Path

    def _init(
        self,
        project_name: str,
        metadata_file: str,
        init_path: pathlib.Path,
        verbose: bool = True,
    ):
        self.project_name = project_name
        self.metadata_file = metadata_file
        self._init_path = init_path
        self.verbose = verbose
        self.path = self._mkdir()

    def experiment_path(self, experiment_name: str) -> "ExperimentPath":
        return ExperimentPath(self, experiment_name)

    def _mkdir(self) -> pathlib.Path:
        home = self._init_path

        path = home / f".cache/{self.project_name}"

        try:
            (path / "experiments").mkdir(exist_ok=True, parents=True)
        except Exception as e:
            if self.verbose:
                print("this error shouldn't have been thrown")
                print(f"error creating path {path}")
                print(e)

        return path

    def metadata_path(self) -> pathlib.Path:
        return (self.path / "experiments") / self.metadata_file


class ProjectCachePath(_ProjectCachePath):
    def __init__(
        self,
        project_name: str,
        metadata_file: str = "metadata.json",
    ):
        home = pathlib.Path.home()
        self._init(project_name, metadata_file, home)


class ExperimentPath:
    project_path: _ProjectCachePath
    experiment_name: str
    path: pathlib.Path
    metadata_file: str

    def __init__(self, project_path: _ProjectCachePath, experiment_name: str):
        self.project_path = project_path
        self.experiment_name = experiment_name

        self.path = self._mkdir()

    def run_path(self, run_name: str) -> "RunPath":
        return RunPath(self, run_name)

    def _mkdir(self) -> pathlib.Path:
        path = self.project_path.path / f"experiments/{self.experiment_name}"

        try:
            path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            if self.project_path.verbose:
                print("this error shouldn't have been thrown")
                print(f"error creating path {path}")
                print(e)
        return path


class RunPath:
    experiment_path: ExperimentPath
    run_name: str

    def __init__(self, experiment_path: ExperimentPath, run_name: str):
        self.experiment_path = experiment_path
        self.run_name = run_name

    def checkpoint_path(self) -> pathlib.Path:
        path = self.experiment_path.path / f"checkpoints/{self.run_name}"

        try:
            path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            if self.experiment_path.project_path.verbose:
                print("this error shouldn't have been thrown")
                print(f"error creating path {path}")
                print(e)
        return path

    def sharded_checkpoint_path(self) -> pathlib.Path:
        path = self.experiment_path.path / f"sharded-checkpoints/{self.run_name}"

        try:
            path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            if self.experiment_path.project_path.verbose:
                print("this error shouldn't have been thrown")
                print(f"error creating path {path}")
                print(e)
        return path

    def lr_scheduler_path(self) -> pathlib.Path:
        path = self.experiment_path.path / f"lr-schedulers/{self.run_name}"

        try:
            path.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            if self.experiment_path.project_path.verbose:
                print("this error shouldn't have been thrown")
                print(f"error creating path {path}")
                print(e)

        return path


def working_directory() -> pathlib.Path:
    return pathlib.Path.cwd()
