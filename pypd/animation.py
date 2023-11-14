"""
Animation class
-----------

"""
from datetime import datetime
import copy

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animation:
    def __init__(self, frequency=100, name=None):
        self.frequency = frequency
        self.name = name or self._generate_animation_name()
        self.frames = []
        self.fig, self.ax = plt.subplots()

    @staticmethod
    def _generate_animation_name():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-animation.gif"

    def save_frame(self, frame):
        """
        Append a matplotlib.figure.Figure object to the self.frames list at a
        frequency defined by self.frequency
        """
        self.frames.append(copy.deepcopy(frame))

    def _set_axis_limits(self, i):
        x_data, y_data = self._get_data_from_frame(i)
        self.ax.set_xlim(min(x_data), max(x_data))
        self.ax.set_ylim(min(y_data), max(y_data))

    def _get_data_from_frame(self, i):
        return self.frames[i].get_axes()[0].collections[0].get_offsets().T

    def _set_scatter_data(self, current_scatter):
        scatter = self.ax.scatter([], [], s=[], c=[], cmap="jet")
        scatter.set_offsets(current_scatter.get_offsets())
        scatter.set_sizes(current_scatter.get_sizes())
        scatter.set_array(current_scatter.get_array())
        return scatter

    def _update(self, frame):
        """
        Update the scatter plot (self.scatter) for each frame in the animation

        Notes
        -----
        Required signature: `def func(frame, *fargs) -> iterable_of_artists`
        """
        self.ax.clear()
        current_scatter = self.frames[frame].get_axes()[0].collections[0]
        self._set_axis_limits(frame)
        scatter = self._set_scatter_data(current_scatter)
        self.ax.set_title(f"frame {frame}")
        self.ax.set_aspect("equal", "box")
        self.ax.axis("off")
        return scatter

    def generate_animation(self):
        """
        Generate an animation from the saved frames (matplotlib.figure.Figure
        objects)

        TODO: it is possible to pass an iterable to frames:
        e.g. frames=self.frames
        """
        self.ani = animation.FuncAnimation(
            self.fig, self._update, frames=len(self.frames), interval=100, blit=False
        )
        self.ani.save(self.name, writer=animation.FFMpegWriter())
