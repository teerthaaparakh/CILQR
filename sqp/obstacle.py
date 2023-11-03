import numpy as np
from params import *


def ellipse_obstacle(E, obs_com, theta, v_obs = 0, t_safe = 0.01):

    a = L/2 + threshold_dist + v_obs*t_safe
    b = W/2 + threshold_dist
    h = obs_com[0]
    k = obs_com[1]
    x = E[0,:]
    y = E[1,:]
    term1 = ((x-h)*np.cos(theta)+(y-k)*np.sin(theta))**2
    term2 = ((x-h)*np.sin(theta)-(y-k)*np.cos(theta))**2
    # print("ter12", a,b,h,k,x,y,theta)
    dist = (term1/a**2 + term2/b**2)-1
    return dist
