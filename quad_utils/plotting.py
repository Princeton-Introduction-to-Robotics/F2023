from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.spatial.transform import Rotation as R
import numpy as np


def _init_animation(drawables: [Line3D]) -> [Line3D]:
    """
    Helper function used by ``animate_quad`` that initializes the drawables for the animation.

    :param drawables: The list of drawables being modified for the current frame. Currently, this is a list of three
           lines representing the two cross bars of the quadrotor and a third line showing the trail of the center of
           mass.
    :return: The list of initialized drawables.
    """

    for d in drawables:
        d.set_xdata(np.array([]))
        d.set_ydata(np.array([]))
        d.set_3d_properties(np.array([]))

    return drawables


def _animate(i: int, states: np.ndarray, drawables: [Line3D]) -> [Line3D]:
    """
    Helper function used by ``animate_quad`` that handles drawing the quadrotor.

    :param i: The frame of the animation being rendered.
    :param states: An 12-by-n matrix, where n is the total number of frames to be rendered. Each column contains a state
           of the quadrotor.
    :param drawables: The list of drawables being modified for the current frame. Currently, this is a list of three
           lines representing the two cross bars of the quadrotor and a third line showing the trail of the center of
           mass.
    :return: The list of modified drawables.
    """
    state = states[:, i]
    wRb = R.from_euler('ZYX', state[[5, 4, 3]] - np.array([0, 0, np.pi / 4]))

    pt1 = state[0:3] + wRb.apply(np.array([1, 0, 0]))
    pt2 = state[0:3] + wRb.apply(np.array([-1, 0, 0]))

    drawables[0].set_xdata(np.array([pt1[0], pt2[0]]))
    drawables[0].set_ydata(np.array([pt1[1], pt2[1]]))
    drawables[0].set_3d_properties(np.array([pt1[2], pt2[2]]))

    pt1 = state[0:3] + wRb.apply(np.array([0, 1, 0]))
    pt2 = state[0:3] + wRb.apply(np.array([0, -1, 0]))

    drawables[1].set_xdata(np.array([pt1[0], pt2[0]]))
    drawables[1].set_ydata(np.array([pt1[1], pt2[1]]))
    drawables[1].set_3d_properties(np.array([pt1[2], pt2[2]]))

    start = int(np.clip(i - 15, 0, i))
    drawables[2].set_xdata(states[0, start:(i + 1)])
    drawables[2].set_ydata(states[1, start:(i + 1)])
    drawables[2].set_3d_properties(states[2, start:(i + 1)])

    return drawables


def animate_quad(timestep: float, states: np.ndarray) -> animation.FuncAnimation:
    """
    Visualize a quadrotor trajectory as an animation. To visualize in a Jupyter notebook, use:

        ``from IPython.display import HTML``
        ``HTML(animate_quad(timestep, states).to_html5_video())``

    Note that ``HTML`` required ``ffmpeg`` to be installed.


    :param timestep: The fixed timestep between each state in the trajectory.
    :param states: An 12-by-n matrix, where each column is a quadrotor state.
    :return: An animation object.
    """
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.grid()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)

    drawables = [ax.plot([], [], [], 'o-r', lw=2)[0],
                 ax.plot([], [], [], 'o-b', lw=2)[0],
                 ax.plot([], [], [],  '-k', lw=2)[0]]

    anim = animation.FuncAnimation(fig, lambda i: _animate(i, states, drawables),
                                   init_func=(lambda: _init_animation(drawables)),
                                   frames=states.shape[1], interval=timestep * 1000, blit=True)

    plt.close(fig)

    return anim
