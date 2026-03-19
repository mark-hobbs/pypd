from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy import spatial

if TYPE_CHECKING:
    from .particles import Particles


class Observation:
    """
    Class for savings observations during a simulation run
    """

    ID_iter: ClassVar[Any] = itertools.count()
    _registry: ClassVar[list[Observation]] = []

    def __init__(
        self,
        coordinates: NDArray[np.float64],
        particles: Particles,
        period: int = 100,
        name: str = "Observation point",
    ) -> None:
        self._registry.append(self)
        self.ID: int = next(Observation.ID_iter)
        self.coordinates: NDArray[np.float64] = coordinates
        _, self.particle = self._nearest_particle(particles)
        self.period: int = period
        self.name: str = name
        self.history: list[NDArray[np.float64]] = []

    def _nearest_particle(self, particles: Particles) -> tuple[float, int]:
        """
        Determine the nearest particle to the user specified observation point

        Parameters
        ----------
        particles : ParticleSet

        Returns
        -------
        distance : float
            Distance between the queried point (observation point) and the
            nearest neighbour

        index : int
            Index of the nearest neighbour

        """
        tree = spatial.KDTree(particles.x)
        d, idx = tree.query(self.coordinates)
        return float(d), int(idx)

    def record_history(
        self, time_step: int, data: NDArray[np.float64]
    ) -> None:
        """
        Record the history of a user defined variable during a simulation run,
        for example, particle displacement
        """
        if time_step % self.period == 0:
            self.history.append(data[self.particle].copy())


class SimulationData:
    """
    Class for saving the output of a simulation run
    """

    def __init__(self) -> None:
        pass

    def record_history(self) -> None:
        """
        Callback that records events into a History (SimulationData) object.

        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History

        The record_history decorator

        https://pythonhosted.org/log_calls/record_history_deco.html
        """
        pass


class History:
    """
    https://bitbucket.org/westmont/history_object/src/master/lib/history_object.py
    """

    def __init__(self, *args: Any) -> None:
        pass

    def start(self, obj: Any) -> None:
        if "_history" not in obj.__dict__:
            obj.__dict__["_history"] = {}

    def __call__(self, cls: type) -> type:
        this = self

        def getter(self: Any, attr: str) -> Any:
            this.start(self)
            if attr == "history":
                return self._history
            if attr == "historyTrace":
                return "\n".join(
                    '%s: "%s" has changed to "%s"' % (t[0], t[1], t[2])
                    for t in self._history
                )
            return self.__dict__.get(attr)

        cls.__getattr__ = getter  # type: ignore[method-assign]

        def setter(self: Any, attr: str, value: Any) -> None:
            this.start(self)
            if self._history.get(attr, False):
                if self._history[attr][-1] == value:
                    pass
                else:
                    self._history[attr].append(value)
            else:
                self._history[attr] = [None, value] if value is not None else [None]
            self.__dict__[attr] = value

        cls.__setattr__ = setter  # type: ignore[method-assign]

        return cls
