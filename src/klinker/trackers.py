# copy-pasted from pykeen
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Pattern, Tuple, Union

from tqdm.auto import tqdm

if TYPE_CHECKING:
    import wandb.wandb_run

__all__ = [
    "ResultTracker",
    "ConsoleResultTracker",
    "WANDBResultTracker",
]


def flatten_dictionary(
    dictionary: Mapping[str, Any],
    prefix: Optional[str] = None,
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    real_prefix = () if prefix is None else (prefix,)
    partial_result = _flatten_dictionary(dictionary=dictionary, prefix=real_prefix)
    return {sep.join(map(str, k)): v for k, v in partial_result.items()}


def _flatten_dictionary(
    dictionary: Mapping[str, Any],
    prefix: Tuple[str, ...],
) -> Dict[Tuple[str, ...], Any]:
    """Help flatten a nested dictionary."""
    result = {}
    for k, v in dictionary.items():
        new_prefix = (*prefix, k)
        if isinstance(v, dict):
            result.update(_flatten_dictionary(dictionary=v, prefix=new_prefix))
        else:
            result[new_prefix] = v
    return result


class ResultTracker:
    """A class that tracks the results from a pipeline run."""

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a run with an optional name.

        Args:
        ----
          run_name: Optional[str]:  (Default value = None)

        Returns:
        -------

        """

    def log_params(
        self, params: Mapping[str, Any], prefix: Optional[str] = None
    ) -> None:
        """Log parameters to result store."""

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log metrics to result store.

        Args:
        ----
          metrics: The metrics to log.
          step: An optional step to attach the metrics to (e.g. the epoch).
          prefix: An optional prefix to prepend to every key in metrics.
        """

    def end_run(self, success: bool = True) -> None:
        """End a run.

        HAS to be called after the experiment is finished.

        Args:
        ----
          success: Can be used to signal failed runs. May be ignored.
        """


class ConsoleResultTracker(ResultTracker):
    """A class that directly prints to console."""

    def __init__(
        self,
        *,
        track_parameters: bool = True,
        parameter_filter: Union[None, str, Pattern[str]] = None,
        track_metrics: bool = True,
        metric_filter: Union[None, str, Pattern[str]] = None,
        start_end_run: bool = False,
        writer: str = "logger",
    ):
        """Initialize the tracker.

        Args:
        ----
            track_parameters: Whether to print parameters.
            parameter_filter: A regular expression to filter parameters. If None, print all parameters.
            track_metrics: Whether to print metrics.
            metric_filter: A regular expression to filter metrics. If None, print all parameters.
            start_end_run: Whether to print start/end run messages.
            writer: The writer to use - one of "tqdm", "builtin", or "logger".
        """
        self.start_end_run = start_end_run

        self.track_parameters = track_parameters
        if isinstance(parameter_filter, str):
            parameter_filter = re.compile(parameter_filter)
        self.parameter_filter = parameter_filter

        self.track_metrics = track_metrics
        if isinstance(metric_filter, str):
            metric_filter = re.compile(metric_filter)
        self.metric_filter = metric_filter

        if writer == "tqdm":
            self.write = tqdm.write
        elif writer == "builtin":
            self.write = print
        elif writer == "logging":
            self.write = logging.getLogger("klinker").info

    def start_run(self, run_name: Optional[str] = None) -> None:
        if run_name is not None and self.start_end_run:
            self.write(f"Starting run: {run_name}")

    def log_params(
        self, params: Mapping[str, Any], prefix: Optional[str] = None
    ) -> None:
        if not self.track_parameters:
            return

        for key, value in flatten_dictionary(dictionary=params).items():
            if not self.parameter_filter or self.parameter_filter.match(key):
                self.write(f"Parameter: {key} = {value}")

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        if not self.track_metrics:
            return
        self.write(f"Step: {step}")
        for key, value in flatten_dictionary(dictionary=metrics, prefix=prefix).items():
            if not self.metric_filter or self.metric_filter.match(key):
                self.write(f"Metric: {key} = {value}")

    def end_run(self, success: bool = True) -> None:
        if not success:
            self.write("Run failed.")
        if self.start_end_run:
            self.write("Finished run.")


class WANDBResultTracker(ResultTracker):
    """A tracker for Weights and Biases.

    Note that you have to perform wandb login beforehand.
    """

    #: The WANDB run
    run: Optional["wandb.wandb_run.Run"]

    def __init__(
        self,
        project: str,
        offline: bool = False,
        **kwargs,
    ):
        """Initialize result tracking via WANDB.

        Args:
        ----
            project:
                project name your WANDB login has access to.
            offline:
                whether to run in offline mode, i.e, without syncing with the wandb server.
            kwargs:
                additional keyword arguments passed to :func:`wandb.init`.

        Raises:
        ------
            ValueError: If the project name is given as None
        """
        import wandb as _wandb

        self.wandb = _wandb
        if project is None:
            raise ValueError("Weights & Biases requires a project name.")
        self.project = project

        if offline:
            os.environ[self.wandb.env.MODE] = "dryrun"  # type: ignore
        self.kwargs = kwargs
        self.run = None

    def start_run(self, run_name: Optional[str] = None) -> None:
        self.run = self.wandb.init(
            project=self.project,
            name=run_name,
            settings=self.wandb.Settings(start_method="fork"),
            **self.kwargs,
        )  # type: ignore

    def end_run(self, success: bool = True) -> None:
        assert self.run
        self.run.finish(exit_code=0 if success else -1)
        self.run = None

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        if self.run is None:
            raise AssertionError("start_run must be called before logging any metrics")
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        self.run.log(metrics, step=step)

    def log_params(
        self, params: Mapping[str, Any], prefix: Optional[str] = None
    ) -> None:
        if self.run is None:
            raise AssertionError("start_run must be called before logging any metrics")
        params = flatten_dictionary(dictionary=params, prefix=prefix)
        self.run.config.update(params)
