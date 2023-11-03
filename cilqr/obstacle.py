from params import *
import numpy as np

def ellipse_obstacle(E, obs_com, theta, v_obs = 0, t_safe = 0.01):

    a = L/2 + threshold_dist + v_obs*t_safe
    b = W/2 + threshold_dist 

    h = obs_com[0]
    k = obs_com[1]

    x = E[0,:]
    y = E[1,:]
    term1 = ((x-h)*np.cos(theta)+(y-k)*np.sin(theta))**2
    term2 = ((x-h)*np.sin(theta)-(y-k)*np.cos(theta))**2

    dist = (term1/a**2 + term2/b**2)-1

    common_term1 = 2*((x-h)*np.cos(theta)+(y-k)*np.sin(theta))
    common_term2 = 2*((x-h)*np.sin(theta)-(y-k)*np.cos(theta))

    #dx
    dterm1_x = common_term1*np.cos(theta)
    dterm2_x = common_term2*np.sin(theta)
    total_term_x = dterm1_x/a**2 + dterm2_x/b**2
    dx = 1/(dist)*(total_term_x)

    #dy
    dterm1_y = common_term1*np.sin(theta)
    dterm2_y = -common_term2*np.cos(theta)
    total_term_y = dterm1_y/a**2 + dterm2_y/b**2
    dy = 1/(dist)*(total_term_y)

    #finding d2x
    dterm1_x2 = dist*(2*np.cos(theta)**2/a**2 + 2*np.sin(theta)**2/b**2)
    dterm2_x2 = total_term_x*(total_term_x)
    d2x = (dterm1_x2 - dterm2_x2)/dist**2

    #finding d2y
    dterm1_y2 = dist*(2*np.sin(theta)**2/a**2 + 2*np.cos(theta)**2/b**2)
    dterm2_y2 = total_term_y*(total_term_y)
    d2y = (dterm1_y2 - dterm2_y2)/dist**2

    #finding dxy
    dterm1_xy = dist*(2*np.sin(theta)*np.cos(theta)/a**2-2*np.sin(theta)*np.cos(theta)/b**2)
    dterm2_xy = total_term_x*(total_term_y)
    dxy = (dterm1_xy - dterm2_xy)/dist**2

    return dist, dx, dy, d2x, d2y, dxy #of size N+1
