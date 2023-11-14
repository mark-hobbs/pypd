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
        self.name = name

        if self.name is None:
            self.name = self._generate_animation_name()
        
        self.frames = []
        self.fig, self.ax = plt.subplots()

        self.scatter = self.ax.scatter([], [], s=[], c=[], cmap="jet")
        self.title = self.ax.set_title('')

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

    def init(self):
        """
        Set up the initial state of the scatter plot
        """        
        # Automatically set axis limits based on the initial data
        x_data = self.frames[0].get_axes()[0].collections[0].get_offsets()[:, 0]
        y_data = self.frames[0].get_axes()[0].collections[0].get_offsets()[:, 1]

        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def update(self, frame):
        """
        Update the scatter plot for each frame in the animation
        """
        self.ax.clear()
        
        # Get the scatter plot from the current figure
        current_scatter = self.frames[frame].get_axes()[0].collections[0]

        # Copy the data from the current scatter plot to the animation scatter plot
        self.scatter = self.ax.scatter([], [], s=[], c=[], cmap="jet")
        self.title = self.ax.set_title('')

        # Automatically set axis limits based on the initial data
        x_data = self.frames[0].get_axes()[0].collections[0].get_offsets()[:, 0]
        y_data = self.frames[0].get_axes()[0].collections[0].get_offsets()[:, 1]

        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        self.scatter.set_offsets(current_scatter.get_offsets())
        self.scatter.set_sizes(current_scatter.get_sizes())
        self.scatter.set_array(current_scatter.get_array())

        self.title.set_text(f'Frame {frame}')

        return self.scatter,

    def generate_animation(self):
        """
        Generate an animation from the saved frames
        
        init_func=self.init,
        blit=True,
        repeat=False,

        extra_args=["-vcodec", "libx264"],
        """
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.frames),
            init_func=self.init,
            interval=100,
            blit=False
        )
        self.ani.save(
            self.name,
            writer=animation.FFMpegWriter()
        )

        # i=0
        # for frame in self.frames:
        #     print(type(frame))
        #     frame.savefig(f"{i}-test.png", dpi=300)
        #     i+=1