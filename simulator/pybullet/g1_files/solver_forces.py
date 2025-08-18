import cvxpy as cp
import numpy as np
import time 
import pinocchio as pin
import pybullet as pb

def solve_force(lambda_val, link_idx_Vec, tau_ext, g1_humanoid, model, data, q_pin, pseudo, printForces):


    if not pseudo:
        Jac = np.array([])

        for link_idx in link_idx_Vec:
            if link_idx == -1:
                frame_id = model.getFrameId(pb.getBodyInfo(g1_humanoid)[0].decode('UTF-8')) 
            else:
                link_name_bytes = pb.getJointInfo(g1_humanoid, link_idx)[12]
                link_name = link_name_bytes.decode('utf-8')
                frame_id = model.getFrameId(link_name)

            J6_worldT = pin.computeFrameJacobian(model, data, q_pin, frame_id, pin.ReferenceFrame.WORLD).T

            Jac = np.hstack((Jac, J6_worldT)) if Jac.size else J6_worldT

        time_start = time.time()

        f_total = cp.Variable(Jac.shape[1])

        error_term = cp.sum_squares(Jac @ f_total - tau_ext)

        l1_regularization = lambda_val * cp.norm(f_total, 1)

        objective = cp.Minimize(error_term + l1_regularization)
        problem = cp.Problem(objective)

        problem.solve()

        time_end = time.time()
        #print(f"Optimization took {time_end - time_start:.4f} seconds")
        if f_total.value is not None:
            estimated_forces = f_total.value.reshape((len(link_idx_Vec), 6))
            if printForces:    
                print(estimated_forces)
            return estimated_forces
       
        raise ValueError("The optimization problem did not return a valid solution.")
        
        # Return de fuerzas en coordenadas mundo, en los indices que indica link_idx_vec

    # if pseudo:

    Jac = np.array([])

    for link_idx in link_idx_Vec:
        if link_idx == -1:
            frame_id = model.getFrameId(pb.getBodyInfo(g1_humanoid)[0].decode('UTF-8')) 
        else:
            link_name_bytes = pb.getJointInfo(g1_humanoid, link_idx)[12]
            link_name = link_name_bytes.decode('utf-8')
            frame_id = model.getFrameId(link_name)

        J6_worldT = pin.computeFrameJacobian(model, data, q_pin, frame_id, pin.ReferenceFrame.WORLD).T

        Jac = np.hstack((Jac, J6_worldT)) if Jac.size else J6_worldT

    time_start = time.time()
    
    f_total = np.linalg.pinv(Jac)@tau_ext
    f_total_reshape = f_total.reshape((len(link_idx_Vec), 6))
    if printForces:
        print(f_total_reshape)
    return f_total_reshape



    