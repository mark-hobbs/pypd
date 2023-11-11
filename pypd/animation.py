"""
Animation class
-----------

"""
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animation:

    def __init__(self, frequency=100, name=None):
        self.frequency = frequency
        self.name = name

        if self.name is None:
            self.name = self._generate_animation_name()
        
        self.frames = []
        self.fig, self.ax = plt.subplots()
        self.sc = self.ax.scatter([], [], s=[], c=[], cmap="jet", alpha=0)

    @staticmethod
    def _generate_animation_name():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-animation.gif"

    def save_frame(self, sc):
        """
        Append a scatter plot to the frames list at a frequency defined by
        self.frequency
        """
        return self.frames.append((sc,))

    def init(self):
        """
        Set up the initial state of the scatter plot
        """
        self.sc.set_offsets([])
        self.sc.set_sizes([])
        self.sc.set_array([])
        return self.sc,

    def update(self, frame):
        """
        Update the scatter plot for each frame in the animation
        """
        updated_plot = self.frames[frame]
        self.sc.set_offsets(updated_plot.get_offsets())
        return (self.sc,)

    def generate_animation(self):
        """
        Generate an animation from the saved frames
        """
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            frames=len(self.frames),
            interval=10,
            blit=True,
            repeat=False,
        )
        self.ani.save(
            self.name,
            writer=animation.FFMpegWriter(),
            fps=1,
            extra_args=["-vcodec", "libx264"],
        )
