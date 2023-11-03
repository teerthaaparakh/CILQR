import numpy as np
from params import *

def obs_loc_and_num_stat():
    if(obs_num_stat):
        obs_loc_stat = np.zeros((obs_num_stat, 2)) # assuming the obs are rectangle
        obs_loc_stat[:,:] = [[20,-0.75], [30, -0.75], [40, -0.75]]
        return obs_loc_stat
    else:
        return None
        # obs_loc_stat[1,:] =
        # obs_loc_stat[2,:] =

def obs_loc_and_num_dyna():
    if(obs_num_dyna):
        obs_loc_dyna_init = np.zeros((obs_num_dyna, 2)) # assuming the obs are rectangle
        obs_loc_dyna_init[:,:] = [[22.5, 0], [0, 4], [52.5,4]]
        obs_speed = np.array([2, 3, 3])
        obs_loc_dyna = np.zeros((obs_num_dyna, 2,N+1))
        obs_loc_dyna[:,:,0] = obs_loc_dyna_init
        obs_loc_dyna[:,1,:] = np.repeat(obs_loc_dyna_init[:,1].reshape(obs_num_dyna,1), N+1, axis = 1)
        for i in range(1, N+1):
            obs_loc_dyna[:,0,i] = obs_loc_dyna[:,0,i-1] + Ts*obs_speed
        return obs_loc_dyna

    else:
        return None

def initial_sequence():
    u_init = np.zeros((2, N))

    init_state = [0,0,0,0]
    ref_traj = np.zeros((2, N+1))
    ref_traj[:,0] = init_state[0:2]
    ref_traj[0,:] = np.arange(N+1)*vel_high*Ts
    return u_init, init_state, ref_traj

def constraints_bounds():
    eq_low = -np.ones((N)*4)*1e-3
    eq_up = np.ones((N)*4)*1e-3

    ineq_low = np.zeros((obs_num_stat + obs_num_dyna+4)*N + N*4)
    ineq_up = np.ones((obs_num_stat + obs_num_dyna +4)*N + N*4)*np.inf

    return eq_low, eq_up, ineq_low, ineq_up
