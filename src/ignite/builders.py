from collections.abc import Callable
from pathlib import Path
import shutil

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Metric


class EngineBuilder:
    """Builder for creating and configuring Ignite engines with dynamic state.

    Example:
        >>> engine = (EngineBuilder(train_step)
        ...     .with_state(model=model, optimizer=optimizer, device=device)
        ...     .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        ...     .with_history(output_transform=lambda out: {"loss": out["loss"]})
        ...     .build())
    """

    def __init__(self, step_function: Callable):
        self._step_function = step_function
        self._state_kwargs: dict[str, object] = {}
        self._metrics: dict[str, Metric] = {}
        self._event_handlers: list = []
        self._history: dict[str, list[float]] = {}

    def with_state(self, **kwargs) -> "EngineBuilder":
        """Add attributes to engine state."""
        self._state_kwargs.update(kwargs)
        return self

    def with_metric(self, name: str, metric: Metric) -> "EngineBuilder":
        """Attach a metric to the engine."""
        self._metrics[name] = metric
        return self

    def with_handler(
        self, event: Events, handler: Callable, *args, **kwargs
    ) -> "EngineBuilder":
        """Add an event handler."""
        self._event_handlers.append((event, handler, args, kwargs))
        return self

    def with_early_stopping(
        self,
        trainer: Engine,
        metric: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        maximize: bool = False,
    ) -> "EngineBuilder":
        """Add early stopping (for validator engines)."""
        sign = 1 if maximize else -1
        handler = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            score_function=lambda engine: sign * engine.state.metrics[metric],
            trainer=trainer,
        )
        return self.with_handler(Events.COMPLETED, handler)

    def with_checkpointing(
        self,
        trainer: Engine,
        checkpoint_dir: Path,
        objects_to_save: dict[str, object],
        metric: str = "loss",
        maximize: bool = False,
        n_saved: int = 1,
        filename_prefix: str = "",
    ) -> "EngineBuilder":
        """Add model checkpointing (for validator engines)."""
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)

        sign = 1 if maximize else -1
        handler = ModelCheckpoint(
            dirname=checkpoint_dir,
            filename_prefix=filename_prefix,
            score_function=lambda engine: sign * engine.state.metrics[metric],
            score_name=metric,
            n_saved=n_saved,
            global_step_transform=lambda engine, _: trainer.state.epoch,
            require_empty=False,
        )
        return self.with_handler(Events.COMPLETED, handler, objects_to_save)

    def with_history(
        self,
        output_transform: Callable[[object], dict[str, float]],
        event: Events = Events.ITERATION_COMPLETED,
    ) -> "EngineBuilder":
        """Collect scalar values per `event` into `self.history`.

        `output_transform(engine.state.output)` must return a flat dict
        `{name: float}`. Each value is appended to `history[name]`.
        """

        def _collect(engine):
            for name, value in output_transform(engine.state.output).items():
                self._history.setdefault(name, []).append(float(value))

        return self.with_handler(event, _collect)

    @property
    def history(self) -> dict[str, list[float]]:
        """Scalars collected via `.with_history(...)`."""
        return self._history

    def build(self) -> Engine:
        """Build and return the configured engine."""
        engine = Engine(self._step_function)
        state_kwargs = self._state_kwargs

        def _inject_state(engine: Engine) -> None:
            for key, value in state_kwargs.items():
                setattr(engine.state, key, value)

        engine.add_event_handler(Events.STARTED, _inject_state)
        for name, metric in self._metrics.items():
            metric.attach(engine, name)
        for event, handler, args, kwargs in self._event_handlers:
            engine.add_event_handler(event, handler, *args, **kwargs)
        return engine
