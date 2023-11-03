import numpy as np
from params import *
from matplotlib import pyplot as plt
import matplotlib as mtl
from colour import Color

def plot(X, obs_loc_stat, obs_loc_dyna):
    light_red = Color("#E5C6BF")# F78066
    colors = list(light_red.range_to(Color("red"),N+1))
    fig, (ax) = plt.subplots(1)

    if obs_num_stat:
        obs_stat = obs_loc_stat.copy()
        obs_stat[:,0] = obs_loc_stat[:,0] - L/2
        obs_stat[:,1] = obs_loc_stat[:,1] - W/2

    if obs_num_dyna:
        obs_dyna = obs_loc_dyna.copy()
        obs_dyna[:,0,:] = obs_loc_dyna[:,0,:] - L/2
        obs_dyna[:,1,:] = obs_loc_dyna[:,1,:] - W/2

    for i in range(obs_num_stat):
        ax.add_patch(mtl.patches.Rectangle((obs_stat[i]),
                                         L, W,
                                         fc = str(colors[-1].hex),ec ='k',lw = 1))

    for i in range(obs_num_dyna):
        for j in range(N+1):
            ax.add_patch(mtl.patches.Rectangle((obs_dyna[i,:,j]),
                                             L, W,
                                             fc = str(colors[j].hex),ec ='k',lw = 1))

    # X_new = X.copy()
    # X_new[0,:] = X[0,:]
    # X_new[1,:] = X[1,:]
    light_green = Color("#D7E5BF")# F78066
    colors = list(light_green.range_to(Color("green"),N+1))

    for j in range(N+1):
        xy = X[:2,j] - np.array([[np.cos(X[3,j]), -np.sin(X[3,j])],[np.sin(X[3,j]), np.cos(X[3,j])]])@np.array([L/2, W/2])
        ax.add_patch(mtl.patches.Rectangle((xy),
                                         L, W, angle = X[3,j]*180/np.pi,
                                         fc = str(colors[j].hex),ec ='k',lw = 1))


    ax.set_ylim([-4,6])
    ax.set_xlim([-4,60])
    ax.set_title("SQP dynamic obstacle")
    ax.set_aspect(1)

    plt.show()
