import numpy as np
from params import *

def get_trajectory(u):
    # u is sequence of steering angle and acceleration of length N
    traj = np.zeros((4, N+1))
    traj[:, 0] = init_state

    for i in range(1, N+1):
        X = traj[:, i-1]
        x0 = X[0]
        y0 = X[1]
        v0 = X[2]
        theta0 = X[3]
        steer = u[0, i-1]
        acc = u[1, i-1]
        kappa = np.tan(steer)/L
        l = v0*Ts + 1/2*acc*Ts**2
        v1 = v0 + acc*Ts
        # print(v1)
        theta1 = theta0 + kappa*l
        if steer:
            x1 = x0 + (np.sin(theta0 + kappa*l) - np.sin(theta0))/kappa
            y1 = y0 + (np.cos(theta0) - np.cos(theta0+ kappa*l))/kappa
        else:
            x1 = x0 + l*np.cos(theta0)
            y1 = y0 + l*np.sin(theta0)

        traj[:,i] = [x1, y1, v1, theta1]

    return traj
