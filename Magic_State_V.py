"""
Code to understand magic states by visualizing single qubit transformations
around the bloch sphere explicitly with the Hadamard and T gate, and with
the Hadamard, T gate, Phase, gate, and Control-Not gate

Code Written by Andrew Projansky
Project Start Date: 7/21/2022
"""

import numpy as np
import matplotlib.pyplot as plt
import random

"""
Defines all universal gates used
"""

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1, 0], [0, 1j]])
T = np.array([[np.exp(-1j * np.pi / 8), 0], [0, np.exp(1j * np.pi / 8)]])
Z = np.array([[1, 0], [0, -1]])
CNot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

"""
Defines all useful states and matrix products for simulation
"""

univ_rot = np.matmul(T, np.matmul(H, np.matmul(T, H)))
magic_state = np.array([1/np.sqrt(2),1/np.sqrt(2)*np.exp(1j*np.pi/4)])
gate_dict = {1: T, 2: H}

class Experiment:
    """
    Parameters
    ----------
    num_steps : int, optional
        total number of steps. The default is 1.
    magic: bool, optional
        Boolean value to determine whether to use magic state or not. The
        default is False
    state: list, optional
        Initial state position. The default is the plus Z eigenstate

    Attributes
    ----------
    angles : list
        angles from states Used for plotting
    """

    def __init__(
        self,
        num_steps: int = 1,
        magic: bool = False,
        state: list = np.array([1,0])

    ):
        self.num_steps = num_steps
        self.magic = magic
        self.state = state


        self.angles = []

    ################# run functions ##################################
    def run_stepwise(self):
        for i in range(self.num_steps):
            gate_choice = random.randint(1,2)
            gate = gate_dict[gate_choice]
            if (gate_choice == 1) and (self.magic):
                magic_step
            else:
                self.state = np.matmul(gate, self.state)
                self.angles.append(self.__get_angle(self.state))

    def magic_steps(self):
        pass

    def __get_angle(self, state):
        """
        get an angle for a single state
        """
        first_mag = np.sqrt(np.real(state[0]) ** 2 + np.imag(state[0]) ** 2)
        second_mag = np.sqrt(np.real(state[1]) ** 2 + np.imag(state[1]) ** 2)
        first_phi = np.arctan2(np.imag(state[0]), np.real(state[0]))
        second_phi = np.arctan2(np.imag(state[1]), np.real(state[1]))
        phi = second_phi - first_phi
        theta = 2 * np.arccos(np.real(state[0] * np.exp(-1j * first_phi)))

        return [theta, phi]

    ##############plot functions ########################
    def plot(self):
        states_xyz = [self.__gen_xyz_points(self.angles)]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.3)
        for x in states_xyz:
            ax.scatter(x[0], x[1], x[2], marker="o", alpha=0.9)
        plt.show()

    def __gen_xyz_points(self, angle_arr):
        """
        Parameters
        ----------
        angle_arr : list
            list of the form [[theta1,phi1],[theta2,phi2] ...].

        """
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for i in angle_arr:
            x = np.append(x, self.__get_x(i[0], i[1]))
            y = np.append(y, self.__get_y(i[0], i[1]))
            z = np.append(z, self.__get_z(i[0], i[1]))

        return x, y, z

    def __get_x(self, theta, phi):
        return np.cos(phi) * np.sin(theta)

    def __get_y(self, theta, phi):
        return np.sin(phi) * np.sin(theta)

    def __get_z(self, theta, phi):
        return np.cos(theta)


#%%
exp = Experiment(num_steps=1000)
exp.run_stepwise()
exp.plot()
del exp
#%%
