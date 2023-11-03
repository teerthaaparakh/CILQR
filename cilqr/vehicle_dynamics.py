from params import *
import numpy as np


def get_trajectory(u, init_state):
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

def forward_simulate(X,U):
    x0 = X[0]
    y0 = X[1]
    v0 = X[2]
    theta0 = X[3]
    steer = U[0]
    acc = U[1]
    kappa = np.tan(steer)/L
    l = v0*Ts + 1/2*acc*Ts**2
    v1 = v0 + acc*Ts
    theta1 = theta0 + kappa*l
    if steer:
        x1 = x0 + (np.sin(theta0 + kappa*l) - np.sin(theta0))/kappa
        y1 = y0 + (np.cos(theta0) - np.cos(theta0+ kappa*l))/kappa
    else:
        x1 = x0 + l*np.cos(theta0)
        y1 = y0 + l*np.sin(theta0)
    return [x1, y1, v1, theta1]


def dynamics_grad(X, u):
    x = X[0]
    y = X[1]
    v = X[2]
    theta = X[3]
    steer = u[0]
    acc = u[1]
    kappa = np.tan(steer)/L
    l = v*Ts + 0.5*acc*Ts**2
    if steer:
        fx = [[1, 0, np.cos(theta+kappa*l)*Ts, (np.cos(theta+kappa*l)-np.cos(theta))/kappa],
            [0, 1, np.sin(theta+kappa*l)*Ts, (np.sin(theta+kappa*l)-np.sin(theta))/kappa],
            [0, 0, 1, 0],
            [0, 0, kappa*Ts, 1]]

        fu = [[np.cos(theta+kappa*l)*l/L*1/np.cos(steer)**2/kappa-
                (np.sin(theta+kappa*l)-np.sin(theta))*1/(L*np.cos(steer)**2*kappa**2),
                    np.cos(theta+kappa*l)*Ts**2/2],
                [np.sin(theta+kappa*l)*l/L*1/np.cos(steer)**2/kappa-
                        (np.cos(theta)-np.cos(theta+kappa*l))*1/(L*np.cos(steer)**2*kappa**2),
                            np.sin(theta+kappa*l)*Ts**2/2],
                [0, Ts],
                [l/L*1/np.cos(steer)**2, kappa/2*Ts**2]]
    else:
        fx = [[1, 0, np.cos(theta)*Ts, -l*np.sin(theta)],
            [0, 1, np.sin(theta)*Ts, l*np.cos(theta)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

        fu = [[0,np.cos(theta)*Ts**2/2],
                [0, np.sin(theta)*Ts**2/2],
                [0, Ts],
                [l/L*1/np.cos(steer)**2, 0]]

    return np.array(fx), np.array(fu)
