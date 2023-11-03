import numpy as np
from params import *
from obstacle import ellipse_obstacle

class obj_and_cons:
    def __init__(self, init_state, obs_loc_stat, obs_loc_dyna, ineq_len, ref_traj):
        self.init_state = init_state
        self.obs_loc_stat = obs_loc_stat
        self.obs_loc_dyna = obs_loc_dyna
        self.ineq_len = ineq_len
        self.ref_traj = ref_traj

    def objective_fun(self, vars):
        """cost of initial state is added to be consistent with CILQR cost,
        but the variables here are X[1,:] and U, X[0,:] is constant, so the cost of
        initial state acts like a constant"""
        cost = np.zeros(N+1)
        u = np.zeros((2,N))
        u[0,:] = vars[0:N]
        u[1,:] = vars[N: N+N]
        X = np.zeros((4, N+1))
        X[:,0] = self.init_state
        X[0,1:] = vars[N*2 : N*2 + N]
        X[1,1:] = vars[N*3 : N*4]
        X[2,1:] = vars[N*4 : N*5]
        X[3,1:] = vars[N*5 : N*6]
        cost[:-1] += wsteer*u[0,:]**2
        cost[:-1] += wacc*u[1,:]**2
        cost += wvel*(X[2,:]-ref_v)**2
        cost += wx*(X[0,:]- self.ref_traj[0,:])**2
        cost += wy*(X[1,:]- self.ref_traj[1,:])**2
        return sum(cost)

    def dynamics_constraint(self,X0,X1,U):
        x0 = X0[0]
        y0 = X0[1]
        v0 = X0[2]
        theta0 = X0[3]

        x1 = X1[0]
        y1 = X1[1]
        v1 = X1[2]
        theta1 = X1[3]

        steer = U[0]
        acc = U[1]

        kappa = np.tan(steer)/L
        l = v0*Ts + 1/2*acc*Ts**2
        v = v1 - (v0 + acc*Ts)
        theta = theta1 - (theta0 + kappa*l)
        if steer:
            x = x1 - (x0 + (np.sin(theta0 + kappa*l) - np.sin(theta0))/kappa)
            y = y1 - (y0 + (np.cos(theta0) - np.cos(theta0 + kappa*l))/kappa)
        else:
            x = x1 - (x0 + l*np.cos(theta0))
            y = y1 - (y0 + l*np.sin(theta0))

        return [x, y, v, theta]

    def equality_contraint(self,vars):
        u = np.zeros((2,N))
        u[0,:] = vars[0:N]
        u[1,:] = vars[N: N+N]
        X = np.zeros((4, N+1))
        X[:,0] = self.init_state
        X[0,1:] = vars[N*2 : N*2 + N]
        X[1,1:] = vars[N*3 : N*4]
        X[2,1:] = vars[N*4 : N*5]
        X[3,1:] = vars[N*5 : N*6]
        cons_list = []

        for i in range(1,N+1):
            X_eq = self.dynamics_constraint(X[:,i-1],X[:,i],u[:,i-1])
            for j in range(4):
                cons_list.append(X_eq[j])
        return cons_list

    def inequality_contraint(self,vars):

        u = np.zeros((2,N))
        u[0,:] = vars[0:N]
        u[1,:] = vars[N: N+N]
        X = np.zeros((4, N+1))
        X[:,0] = self.init_state
        X[0,1:] = vars[N*2 : N*2 + N]
        X[1,1:] = vars[N*3 : N*4]
        X[2,1:] = vars[N*4 : N*5]
        X[3,1:] = vars[N*5 : N*6]
        cons_list = np.zeros(self.ineq_len)

        cons_list[0 : N] = u[0,:]-steer_low-factor #1
        cons_list[N : N*2] = steer_high-factor-u[0,:] #1

        cons_list[N*2 : N*3] = u[1,0:]-acc_low-factor #1
        cons_list[N*3 : N*4] = acc_high-factor-u[1,0:] #1

        cons_list[N*4 : N*5] = X[1,1:]-ylim_low-factor #1
        cons_list[N*5 : N*6] = ylim_high-factor-X[1,1:] #1

        cons_list[N*6 : N*7] = X[2,1:]-vel_low-factor #1
        cons_list[N*7 : N*8] = vel_high-factor-X[2,1:] #1

        for j in range(obs_num_stat): #obs_num_stat
            dist = ellipse_obstacle(X[:2,1:], self.obs_loc_stat[j,:], 0)
            cons_list[N*8 +j*(N): N*9+j*(N)] = dist - 0.5

        index = N*8+(obs_num_stat)*(N)
        for j in range(obs_num_dyna):

            dist = ellipse_obstacle(X[:2,1:], self.obs_loc_dyna[j,:,1:], 0)
            cons_list[ index+j*(N) : index + (j+1)*(N)] = dist - 0.5


        return cons_list
