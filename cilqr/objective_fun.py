from params import *
from obstacle import ellipse_obstacle
import numpy as np

def objective_fun_and_grad(u, X, t, ref_traj = None, obs_loc_stat = None, obs_loc_dyna = None):
    cost = np.zeros(N+1)
    Lx = np.zeros((4, N+1))
    Lxx = np.zeros((4, 4, N+1))
    Lu = np.zeros((2, N))
    Luu = np.zeros((2, 2, N,))

    cost[:-1] += wsteer*u[0,:]**2
    cost[:-1] += wacc*u[1,:]**2
    cost += wvel*(X[2,:]-ref_v)**2
    cost += wx*(X[0,:]- ref_traj[0,:])**2
    cost += wy*(X[1,:]- ref_traj[1,:])**2

    #following inequalities added to objective as penalties
    cost += (-1/t)*np.log(-(-X[1,:]+ylim_low))
    cost += (-1/t)*np.log(-(X[1,:]-ylim_high))

    cost += (-1/t)*np.log(-(-X[2,:]+vel_low))
    cost += (-1/t)*np.log(-(X[2,:]-vel_high))

    cost[:-1] += (-1/t)*np.log(-(-u[0,:]+steer_low))
    cost[:-1] += (-1/t)*np.log(-(u[0,:]-steer_high))

    cost[:-1] += (-1/t)*np.log(-(-u[1,:]+acc_low))
    cost[:-1] += (-1/t)*np.log(-(u[1,:]-acc_high))

    Dx = np.zeros(N+1)
    Dy = np.zeros(N+1)
    D2x = np.zeros(N+1)
    D2y = np.zeros(N+1)
    Dxy = np.zeros(N+1)

    for j in range(obs_num_stat):
        dist, dx, dy, d2x, d2y, dxy = \
            ellipse_obstacle(X[:2,:], obs_loc_stat[j,:], 0)
        cost += (-1/t)*np.log(dist)

        Dx += dx
        Dy += dy
        D2x += d2x
        D2y += d2y
        Dxy += dxy

#    obs_dyna = obs_loc_dyna if obs_num_dyna else 0
    for j in range(obs_num_dyna):
        dist, dx, dy, d2x, d2y, dxy = \
            ellipse_obstacle(X[:2,:], obs_loc_dyna[j,:,:], 0)

        cost += (-1/t)*np.log(dist)
        Dx += dx
        Dy += dy
        D2x += d2x
        D2y += d2y
        Dxy += dxy
    zeros_array = np.zeros(N+1)

    Lx[:,:] = [(-1/t)*Dx +2*wx*(X[0,:]-ref_traj[0,:]), \
        2*wy*(X[1,:]-ref_traj[1,:]) + (-1/t)*(1)/(X[1,:]-ylim_low) + (-1/t)*(1)/(X[1,:]-ylim_high) + (-1/t)*Dy,\
         2*wvel*(X[2,:]-ref_v) + (-1/t)*(1)/(X[2,:]-vel_low) + (-1/t)*(1)/(X[2,:]-vel_high) ,\
          zeros_array]
    Lxx[:,:,:] = [[(-1/t)*D2x + 2*wx, -1/t*Dxy, zeros_array, zeros_array],
            [-1/t*Dxy, 2*wy + (-1/t)*D2y + (-1/t)*(-1)/(X[1,:]-ylim_low)**2 + (-1/t)*(-1)/(X[1,:]-ylim_high)**2, zeros_array, zeros_array],
            [zeros_array, zeros_array, 2*wvel + (-1/t)*(-1)/(X[2,:]-vel_low)**2 + (-1/t)*(-1)/(X[2,:]-vel_high)**2, zeros_array],
            [zeros_array, zeros_array, zeros_array, zeros_array]]

    Lu[:,:] = [wsteer*u[0,:]*2 + (-1/t)*(1)/(u[0,:]-steer_low) \
            + (-1/t)*(1)/(u[0,:]-steer_high), \
        wacc*u[1,:]*2 + (-1/t)*(1)/(u[1,:]-acc_low) \
                    + (-1/t)*(1)/(u[1,:]-acc_high)]

    Luu[:,:,:] = [[wsteer*2 + (-1/t)*(-1)/(u[0,:]-steer_low)**2 \
            + (-1/t)*(-1)/(u[0,:]-steer_high)**2, np.zeros(N)],
            [np.zeros(N), wacc*2 + (-1/t)*(-1)/(u[1,:]-acc_low)**2 \
                    + (-1/t)*(-1)/(u[1,:]-acc_high)**2]]

    return cost, Lx, Lxx, Lu, Luu
