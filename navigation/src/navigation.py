import numpy as np 
from math import sqrt, cos, sin, atan2, pi, degrees  

class Navigation:

    def __init__(self):

        # Dynamic parameters
        self.m = 28.20                           # mass of heron [kg]
        self.Izz = 10.04                         # mass moment of inertia [kg-m^2]
        self.M = np.array([[self.m, 0,       0], # inertia matrix
                            [0,     self.m,  0],
                            [0,     0,       self.Izz]])   
        self.D = np.identity(3)                  # damping matrix 
        self.u = np.zeros(3)                     # control input (Twist) 

        # State variables
        self.state = np.zeros(6)
        self.state_hat = np.zeros(6)
        self.prevState = np.zeros(6)

        # System Matrices  
        self.B = np.identity(6)                # control transition matrix
        self.P = np.identity(6)                # process noise covariance matrix
        self.P_hat = np.identity(6)            # estimated process noise covariance matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], # observation matrix 
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])      
        self.Q = np.identity(np.size(self.state))                # sensor measurement noise
        self.K = 0                             # Kalman gain 
        self.Z = np.zeros(6)                   # sensor measurements

        # Other variables
        self.gpsData = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        self.imuData = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        self.bodyECEF = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.yaw = 0.0
        self.dt = 1

    def set_gps_data(self, gpsData):
        self.gpsData = gpsData
        self.update_sensor_measurements()

    def set_gps_covariance(self, gpsCov):
        self.Q[0][0] = gpsCov[0]
        self.Q[1][1] = gpsCov[4]

    def set_imu_data(self, imuData):
        self.imuData = imuData
        self.update_sensor_measurements()

    def set_yaw(self, yaw):
        self.yaw = yaw
        self.update_sensor_measurements()
    
    def set_cmd_input(self, cmd_input):
        self.u = cmd_input

    def update_sensor_measurements(self):
        self.Z[0], self.Z[1], self.Z[2] = self.gps2ecef(self.gpsData['lat'], self.gpsData['lon'], self.gpsData['alt'])

    def R_matrix(self, psi):
        R = np.array([ [cos(psi), -sin(psi), 0],
                       [sin(psi), cos(psi),  0],
                       [0,        0,         1]])
        return R

    def get_initial_state(self):
        '''
        Initalizes initial state for state estimation. 
        '''
        # Convert GPS to ECEF
        self.bodyECEF['x'], self.bodyECEF['y'], self.bodyECEF['z'] = self.gps2ecef(self.gpsData['lat'], self.gpsData['lon'], self.gpsData['alt'])

        self.state[0] = self.bodyECEF['x'] 
        self.state[1] = self.bodyECEF['y']
        self.state[2] = self.yaw
        self.state[3] = 0.
        self.state[4] = 0.
        self.state[5] = 0.

    def predict(self):
        # Update the A & B matrices
        self.update_A_matrix()
        self.update_B_matrix()
        # Predict with system model 
        self.state_hat = np.dot(self.A, self.state) + np.dot(self.B, self.u)
        self.P_hat = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update_A_matrix(self):
        A1 = self.R_matrix(self.state[2])
        A2 = -1 * np.dot(self.M, self.D)
        A3 = np.zeros((3,3))
        A4 = np.hstack((A3,A1))
        A5 = np.hstack((A3,A2))
        self.A = np.vstack((A4,A5))

    def update_B_matrix(self):
        B1 = self.R_matrix(self.state[2])
        B2 = np.array([[(self.state[3]-self.prevState[3])/self.dt, 0, 0],
                        [0, (self.state[4]-self.prevState[4])/self.dt, 0],
                        [0, 0, (self.state[5]-self.prevState[5])/self.dt]])
        self.B = np.vstack((B1,B2))

    def calculateKalmanGain(self):
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv((np.dot(self.H, np.dot(self.P,self.H.T)) + self.Q)))  

    def update(self):
        self.prevState = self.state 
        self.P = np.dot((np.identity(6) - np.dot(self.K,self.H)), self.P_hat)
        self.state = self.state_hat + np.dot(self.K, (self.Z - np.dot(self.H, self.state_hat)))
        self.state[0], self.state[1], self.gpsData['z'] = self.ecef2gps(self.state[0], self.state[1], self.bodyECEF['z'])

    def update_state(self):
        self.predict()
        self.calculateKalmanGain()
        self.update()
        return self.state

    def gps2ecef(self, lat, lon, alt):
        rad_lat = lat * (pi / 180.0)
        rad_lon = lon * (pi / 180.0)

        a = 6378137.0
        finv = 298.257223563
        f = 1 / finv
        e2 = 1 - (1 - f) * (1 - f)
        v = a / sqrt(1 - e2 * sin(rad_lat) * sin(rad_lat))

        x = (v + alt) * cos(rad_lat) * cos(rad_lon)
        y = (v + alt) * cos(rad_lat) * sin(rad_lon)
        z = (v * (1 - e2) + alt) * sin(rad_lat)

        return x, y, z

    def ecef2gps(self, x, y, z):
        a = 6378137.0 #in meters
        b = 6356752.314245 #in meters

        f = (a - b) / a
        f_inv = 1.0 / f

        e_sq = f * (2 - f)                       
        eps = e_sq / (1.0 - e_sq)

        p = sqrt(x * x + y * y)
        q = atan2((z * a), (p * b))

        sin_q = sin(q)
        cos_q = cos(q)

        sin_q_3 = sin_q * sin_q * sin_q
        cos_q_3 = cos_q * cos_q * cos_q

        phi = atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3))
        lam = atan2(y, x)

        v = a / sqrt(1.0 - e_sq * sin(phi) * sin(phi))
        h   = (p / cos(phi)) - v

        lat = degrees(phi)
        lon = degrees(lam)

        return lat, lon, h
