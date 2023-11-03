import numpy as np
from params import *
from obj_and_cons import obj_and_cons
from initialize import obs_loc_and_num_stat, obs_loc_and_num_dyna, initial_sequence, constraints_bounds
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from plotting import plot
import time

if __name__ == "__main__":
    obs_loc_stat = obs_loc_and_num_stat()
    obs_loc_dyna = obs_loc_and_num_dyna()
    u_init, init_state, ref_traj = initial_sequence()
    eq_low, eq_up, ineq_low, ineq_up = constraints_bounds()

    obj_and_cons_class = obj_and_cons(init_state, obs_loc_stat, obs_loc_dyna, ineq_low.shape[0], ref_traj)

    eq_nonlinear_constraint = NonlinearConstraint(obj_and_cons_class.equality_contraint, eq_low, eq_up)
    ineq_nonlinear_constraint = NonlinearConstraint(obj_and_cons_class.inequality_contraint, ineq_low, ineq_up)
    x0 = np.zeros(N*2 + (N)*4)
    start = time.process_time()
    res = minimize(obj_and_cons_class.objective_fun, x0, method='SLSQP', constraints=[eq_nonlinear_constraint, ineq_nonlinear_constraint],\
                            options={'maxiter': 200})
    print("time taken", time.process_time() - start)
    print("cost ", res.fun)
    print("iterations ", res.nit)

    ans = res.x
    u = np.zeros((2,N))
    u[0,:] = ans[0:N]
    u[1,:] = ans[N: N+N]
    X = np.zeros((4, N+1))
    X[0,1:] = ans[N*2 : N*2 + N]
    X[1,1:] = ans[N*3 : N*4]
    X[2,1:] = ans[N*4 : N*5]
    X[3,1:] = ans[N*5 : N*6]

    plot(X, obs_loc_stat, obs_loc_dyna)
