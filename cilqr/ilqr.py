from params import *
from vehicle_dynamics import dynamics_grad, forward_simulate, get_trajectory
from objective_fun import objective_fun_and_grad
from constraints_check import constraints_check
import numpy as np

def backward_pass(Lx, Lxx, Lu, Luu, traj, U, rho):
    Vkx = Lx[:,-1]
    Vkxx = Lxx[:, :, -1]
    Du_star = np.zeros((2,N))
    q = np.zeros((2,N))
    Q = np.zeros(( 2,4,N))

    for i in range(N-1, -1, -1):
        fx, fu = dynamics_grad(traj[:,i], U[:,i])

        Px = Lx[:, i] + Vkx@fx
        Pu = Lu[:, i] + Vkx@fu
        Pxx = Lxx[:, :, i] + fx.T@Vkxx@fx
        Puu = Luu[:, :, i] + fu.T@(Vkxx + rho*np.eye(4))@fu
        Pux = 0 + fu.T@(Vkxx + rho*np.eye(4))@fx
        Pxu = Pux.T #0 + fx.T@(Vkxx + rho*np.eye(4))@fu
        qi = -np.linalg.inv(Puu )@Pu
        Qi = -np.linalg.inv(Puu )@Pux
        q[:,i] = qi
        Q[:,:, i] = Qi

        Vkx = Px + Qi.T.dot(Puu).dot(qi)
        Vkx += Qi.T.dot(Pu) + Pux.T.dot(qi)

        Vkxx = Pxx + Qi.T.dot(Puu).dot(Qi)
        Vkxx += Qi.T.dot(Pux) + Pux.T.dot(Qi)
        # Vkxx = 0.5 * (Vkxx + Vkxx.T)  # To maintain symmetry.
    return q, Q

def forward_pass(q, Q, traj_old, U_old, alpha = 1):
    traj_new = np.zeros((4, N+1))
    traj_new[:,0] = traj_old[:,0]
    U_new = np.zeros((2, N))
    for i in range(1,N+1):
        U_new[:,i-1] = U_old[:,i-1] + alpha*(q[:,i-1] + Q[:,:, i-1]@(traj_new[:,i-1]-traj_old[:,i-1]))
        traj_new[:,i] = forward_simulate(traj_new[:,i-1], U_new[:,i-1])
    return traj_new, U_new

def combined(u_init, init_state, ref_traj = None, obs_loc_stat = None, obs_loc_dyna = None):
    u_star = u_init
    traj = get_trajectory(u_star, init_state)
    t = 50
    total_iter = 0

    cost_outer_prev = np.inf

    cost_outer_curr, Lx, Lxx, Lu, Luu = objective_fun_and_grad(u_star, traj, t, ref_traj, obs_loc_stat, obs_loc_dyna)
    cost_outer_curr = sum(cost_outer_curr)

    while abs(cost_outer_prev - cost_outer_curr) > 1e-2: #outer loop
        # print("outer cost", cost_outer_prev - cost_outer_curr)
        total_iter += 1
        rho = 0
        changed = True
        cost_inner_prev = np.inf
        cost_inner_curr, Lx, Lxx, Lu, Luu = objective_fun_and_grad(u_star, traj, t, ref_traj, obs_loc_stat, obs_loc_dyna)
        cost_inner_curr = sum(cost_inner_curr)

        while abs(cost_inner_prev- cost_inner_curr) > 1e-2:
            # print("inner cost before starting", cost_inner_prev- cost_inner_curr)
            q, Q = backward_pass(Lx, Lxx, Lu, Luu, traj, u_star, rho = rho) #inner loop
            traj_new, u_star_new = forward_pass(q, Q, traj, u_star)
            alpha = 0.5
            line_search_num = 0
            while  (line_search_num < 10) and (not constraints_check(u_star_new, traj_new, obs_loc_stat, obs_loc_dyna)):
                alpha *= backtrack_factor
                traj_new, u_star_new = forward_pass(q, Q, traj, u_star, alpha)
                line_search_num += 1
            if(line_search_num == 10 and (not constraints_check(u_star_new, traj_new, obs_loc_stat, obs_loc_dyna))):
                rho += 0.01
                changed = False
            else:
                traj = traj_new
                u_star = u_star_new
                rho = 0
                changed = True
                cost_inner_prev = cost_inner_curr
                cost_inner_curr, Lx, Lxx, Lu, Luu= objective_fun_and_grad(u_star, traj, t, ref_traj, obs_loc_stat, obs_loc_dyna)
                cost_inner_curr = sum(cost_inner_curr)

            if(rho>0.5):
                # print("solution not found !!")
                return cost_inner_curr, traj, u_star, total_iter, False

        # print("inner cost after exiting", cost_inner_prev- cost_inner_curr)
        # print("\n")
        cost_outer_prev = cost_outer_curr
        cost_outer_curr = cost_inner_curr
        t *= mu
    # print("solution found !!")
    # print("total iterations", total_iter)
    return cost_inner_curr, traj, u_star, total_iter, True
