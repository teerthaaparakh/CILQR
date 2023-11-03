from params import *
from obstacle import ellipse_obstacle
import numpy as np


def constraints_check(u, X, obs_loc_stat = None, obs_loc_dyna = None):
    cons = np.any(u[0,:] < steer_low)
    if(cons):
        # print("constraints not satsified steer low")
        return False

    cons = np.any(u[0,:] > steer_high)
    if(cons):
        # print("constraints not satsified steer_high")
        return False

    cons = np.any(u[1,:] < acc_low)
    if(cons):
        # print("constraints not satsified acc_low")
        return False

    cons = np.any(u[1,:] > acc_high)
    if(cons):
        # print("constraints not satsified acc_high")
        return False
    
    cons = np.any(X[2,:] < vel_low)
    if(cons):
        # print("constraints not satsified vel_low")
        return False

    cons = np.any(X[2,:] > vel_high)
    if(cons):
        # print("constraints not satsified vel_high")
        return False

    #or steer_high or acc_low or acc_high or vel_low or vel_high or ylim_low or ylim_high):

    cons = np.any(X[1,:] < ylim_low)
    if(cons):
        # print("constraints not satsified ylim_low")
        return False

    cons = np.any(X[1,:] > ylim_high)
    if(cons):
        # print("constraints not satsified ylim_high")
        return False

    for j in range(obs_num_stat):
        min_dist, _, _, _, _, _ = \
            ellipse_obstacle(X[:2,:], obs_loc_stat[j,:], 0)
        if np.any(min_dist<=0):
            return False

#    obs_loc_dyna = obs_loc_dyna if obs_num_dyna else 0
    for j in range(obs_num_dyna):
        min_dist, _, _, _, _, _ = \
            ellipse_obstacle(X[:2,:], obs_loc_dyna[j, :, :], 0)
        if np.any(min_dist<=0):
            return False
    return True
