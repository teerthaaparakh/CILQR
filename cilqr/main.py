from ilqr import combined
from initialize import obs_loc_and_num_stat, obs_loc_and_num_dyna, initial_sequence
from plotting import plot
import numpy as np
import time


if __name__ == "__main__":
    obs_loc_stat = obs_loc_and_num_stat()
    obs_loc_dyna = obs_loc_and_num_dyna()
    u_init, init_state, ref_traj = initial_sequence()

    start = time.process_time()
    cost_curr, traj, u_star, niter, success = combined(u_init, init_state, ref_traj, obs_loc_stat, obs_loc_dyna)
    if(success):
        print("solution found !!")
    else:
        print("no solution found !!")
    print("time taken ", time.process_time() - start)
    print("cost ", cost_curr)
    print("total iterations ", niter)
    plot(traj, obs_loc_stat, obs_loc_dyna)
