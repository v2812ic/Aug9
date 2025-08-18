import numpy as np
import pinocchio as pin

from config.g1.sim.pybullet.ihwbc.pybullet_params import *

# ----- parámetros -----------------------------------------------------------
n_filter = Config.n_filter_observer    
ki = Config.gain_observer         

if not Config.observerActive:
    ki = 0

K = [ki*np.eye(33) for _ in range(n_filter)]
gamma = np.zeros((n_filter, 33))    
k_prev = np.zeros(33)             
m_prev = np.zeros(33)                
k_prev_2 = np.zeros(33)

def get_tau_ext(dt, model, data, q, v, tau_c, tau_j):

    global gamma, k_prev, m_prev

    pin.computeAllTerms(model, data, q, v)
    M, C, g = data.M, data.C, data.g

    k = M @ v
    m = C.T @ v - g + tau_j + tau_c 
    
    if n_filter == 0:
        raise ValueError("El orden del filtro no puede ser ")

    gamma[0] = (K[0] @ (k - k_prev - m*dt)      
                + (np.eye(33) - dt*K[0]) @ gamma[-1])
    
    for i in range(1, n_filter):
        gamma[i] += K[i] @ (-gamma[-1] + gamma[i-1]) *dt
    
    k_prev = k

    return gamma[-1]       
    