"""
Animation class
-----------

"""
from datetime import datetime


class Animation:

    def __init__(self, simulation, frequency=100, name=None) -> None:
        self.frequency = frequency
        self.n_frames = frequency / simulation.n_time_steps
        self.frames = []
        self.name = name

        if self.name is None:
            self.name = self._generate_animation_name()

    @staticmethod
    def _generate_animation_name():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-animation.gif"

    def save_frame(self):
        """
        Save a single frame
        """
        pass

    def generate_animation(self):
        """
        Generate an animation from the saved frames
        """
        pass