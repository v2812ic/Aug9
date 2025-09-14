import os, numpy as np, datetime

class ComLogger:
    def __init__(self, outdir="experiment_data", basename="com_real"):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.basename = basename

    def dump(self, time, com_pos, com_vel, com_acc=None):
        t = np.asarray(time, dtype=float)
        C = np.asarray(com_pos, dtype=float)
        V = np.asarray(com_vel, dtype=float)
        A = np.asarray(com_acc, dtype=float) if (com_acc is not None and len(com_acc)>0) else None
        stem = os.path.join(self.outdir, f"{self.basename}_{0}")

        # 1) .mat (ideal para MATLAB)
        try:
            from scipy.io import savemat
            mdic = {"t_act": t, "com_pos_act": C, "com_vel_act": V}
            if A is not None: mdic["com_acc_act"] = A
            savemat(stem + ".mat", mdic)
            print(f"[ComLogger] Guardado {stem}.mat")
            return
        except Exception as e:
            print(f"[ComLogger] No pude escribir .mat ({e}).")

