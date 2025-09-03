import numpy as np
from scipy.interpolate import griddata
import os

class PressureInterpolator:
    def __init__(self, data_path="./config/g1/sim/pybullet/ihwbc/parsed_drag_data.csv"):
        
        try:
            data = np.loadtxt(data_path, delimiter = ',', skiprows = 1)

            self.points = data[:, :2]
            self.values = data[:, 2]

            print("Drag interpolator loaded")
        
        except Exception as e:
            print(f"Error loading drag data: {e}")

    def get_cd_from_distribution(self, yaw_angle_deg, reynolds_normal):

        query_point = np.array([[yaw_angle_deg, reynolds_normal]])

        cd_value = griddata(self.points, self.values, query_point, method = 'linear')

        if np.isnan(cd_value):
            print(f"Warning: Extrapolation for angle={yaw_angle_deg} and Re={reynolds_normal}, using nearest")
            cd_value = griddata(self.points, self.values, query_point, method = 'nearest')

        return cd_value.flatten()