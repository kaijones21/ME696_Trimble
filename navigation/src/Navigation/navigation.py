import numpy as np 
from math import sqrt, cos, sin, atan2, pi  

class Navigation:

    def __init__(self):

        # Dynamic parameters
        self.M = np.identity(3)              # inertia matrix
        self.D = np.identity(3)              # damping matrix 
        self.tau = np.array([0.0, 0.0, 0.0]) # force vector  

        # State variables
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.stateHat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prevTime = 0.0

        # System Matrices 
        self.A = np.identity(6)     # state transition matrix  
        self.B = np.identity(6)     # control transition matrix
        self.P = np.identity(6)     # process noise covariance matrix
        self.P_hat = np.identity(6) # estimated process noise covariance matrix
        self.H = np.identity(6)     # sensor transition matrix 
        self.Q = np.identity(6)     # sensor measurement noise

        # Other variables
        self.gpsData = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        self.imuData = {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0}
        self.yaw = 0.0

    def set_gps_data(self, gpsData):
        self.gpsData = gpsData

    def set_imu_data(self, imuData):
        self.imuData = imuData

    def set_yaw(self, yaw):
        self.yaw = yaw

    def R_matrix(self, psi):
        R = np.array([ [cos(psi), -sin(psi), 0],
                       [sin(psi), cos(psi),  0],
                       [0,        0,         1]])
        return R

    def get_initial_state(self):
        '''
        Initalizes initial state for state estimation. 

            Parameters:
                bodyGPS (dict): contains 'lat', 'lon', and 'alt' of body frame
                mapGPS (dict): contains 'lat', 'lon', and 'alt' of map frame
                imu (dict): contains heading of body frame w.r.t map frame
        '''

        # Assume initial velocities is 0
        self.state[3] = 0.0
        self.state[4] = 0.0
        self.state[5] = 0.0

        # Convert GPS to ECEF
        mapECEF = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        bodyECEF = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        mapECEF['x'], mapECEF['y'], mapECEF['z'] = self.gps2ecef(self.gpsData['lat'], self.gpsData['lon'], self.gpsData['alt'])
        bodyECEF['x'], bodyECEF['y'], bodyECEF['z'] = self.gps2ecef(self.gpsData['lat'], self.gpsData['lon'], self.gpsData['alt'])
        self.state[0] = bodyECEF['x'] - mapECEF['x']
        self.state[1] = bodyECEF['y'] - mapECEF['y']

        # Get yaw
        self.state[2] = self.yaw
        print(self.state)

    def predict(self):
        x_hat = np.dot(self.A, self.state) + np.dot(self.B, self.u)
        P_hat = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return (x_hat, P_hat)

    def calculateKalmanGain(P, H, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv((np.dot(H, np.dot(P,H.T)) + R)))  
        print("Kalman Gain: ")
        print(K)
        return K 

    def update(K, H, P_hat, x_hat, Y): 
        P = np.dot((np.identity(3) - np.dot(K,H)), P_hat)
        x = x_hat + np.dot(K, (Y - np.dot(H, x_hat)))
        return (x, P)

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
