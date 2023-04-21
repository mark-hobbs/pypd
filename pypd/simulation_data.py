import numpy as np
import itertools
from scipy import spatial


class Observation:
    """
    Class for savings observations during a simulation run
    """

    ID_iter = itertools.count()
    _registry = []

    def __init__(self, coordinates, particles, period=100, name="Observation point"):
        self._registry.append(self)
        self.ID = next(Observation.ID_iter)
        self.coordinates = coordinates
        _, self.particle = self._nearest_particle(particles)
        self.period = period
        self.name = name
        self.history = []

    def _nearest_particle(self, particles):
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
        return tree.query(self.coordinates)

    def record_history(self, time_step, data):
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

    def __init__(self):
        pass

    def record_history(self):
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

    def __init__(self, *args):
        pass

    def start(self, obj):
        if "_history" not in obj.__dict__:
            obj.__dict__["_history"] = {}

    def __call__(self, cls):
        this = self

        def getter(self, attr):
            this.start(self)
            if attr == "history":
                return self._history
            if attr == "historyTrace":
                return "\n".join(
                    '%s: "%s" has changed to "%s"' % (t[0], t[1], t[2])
                    for t in self._history
                )
            return self.__dict__.get(attr)

        cls.__getattr__ = getter

        def setter(self, attr, value):
            this.start(self)
            if self._history.get(attr, False):
                if self._history[attr][-1] == value:
                    pass
                else:
                    self._history[attr].append(value)
            else:
                self._history[attr] = [None, value] if value is not None else [None]
            self.__dict__[attr] = value

        cls.__setattr__ = setter

        return cls
