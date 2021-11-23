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
        self.state_hat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.prev_time = 0.0

        # System Matrices 
        self.A = np.identity(6)     # state transition matrix  
        self.B = np.identity(6)     # control transition matrix
        self.P = np.identity(6)     # process noise covariance matrix
        self.P_hat = np.identity(6) # estimated process noise covariance matrix
        self.H = np.identity(6)     # sensor transition matrix 
        self.Q = np.identity(6)     # sensor measurement noise
         
    def R_matrix(self, psi):
        R = np.array([ [cos(psi), -sin(psi), 0],
                       [sin(psi), cos(psi),  0],
                       [0,        0,         1]])
        return R

    def get_initial_state(self, bodyGPS, mapGPS, yaw):
        '''
        Initalizes initial state for state estimation. 

            Parameters:
                bodyGPS (dict): contains 'lat', 'lon', and 'alt' of body frame
                mapGPS (dict): contains 'lat', 'lon', and 'alt' of map frame
                imu (dict): contains heading of body frame w.r.t map frame
        '''
        
        # Assume initial velocities is 0
        self.state(3) = 0.0
        self.state(4) = 0.0
        self.state(5) = 0.0

        # Convert GPS to ECEF
        mapECEF = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        bodyECEF = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        mapECEF['x'], mapECEF['y'], mapECEF['z'] = self.gps_to_ecef_custom(mapGPS['lat'], mapGPS['lon'], mapGPS['alt'])
        bodyECEF['x'], bodyECEF['y'], bodyECEF['z'] = self.gps_to_ecef_custom(bodyGPS['lat'], bodyGPS['lon'], bodyGPS['alt'])
        self.state[0] = bodyECEF['x'] - mapECEF['x']
        self.state[1] = bodyECEF['y'] - mapECEF['y']

        # Get yaw
        self.state[3] = yaw

    def predict(A, B, x, u, Q, P):
        x_hat = np.dot(A, x) + np.dot(B, u)
        P_hat = np.dot(A, np.dot(P, A.T)) + Q
        return (x_hat, P_hat)

    def calculateKalmanGain(P, H, R):
        K = np.dot(np.dot(P, H.T), np.linalg.inv((np.dot(H, np.dot(P,H.T)) + R)))  
        return K 

    def update(K, H, P_hat, x_hat, Y): 
        P = np.dot((np.identity(3) - np.dot(K,H)), P_hat)
        x = x_hat + np.dot(K, (Y - np.dot(H, x_hat)))
        return (x, P)

coords = [
  (37.4001100556,  -79.1539111111,  208.38),
  (37.3996955278,  -79.153841,  208.48),
  (37.3992233889,  -79.15425175,  208.18),
  (37.3989114167,  -79.1532775833,  208.48),
  (37.3993285556,  -79.1533773333,  208.28),
  (37.3992801667,  -79.1537883611,  208.38),
  (37.3992441111,  -79.1540981944,  208.48),
  (37.3992616389,  -79.1539428889,  208.58),
  (37.3993530278,  -79.1531711944,  208.28),
  (37.4001223889,  -79.1538085556,  208.38),
  (37.3992922222,  -79.15368575,  208.28),
  (37.3998074167,  -79.1529132222,  208.18),
  (37.400068,  -79.1542711389,  208.48),
  (37.3997516389,  -79.1533794444,  208.38),
  (37.3988933333,  -79.1534320556,  208.38),
  (37.3996279444,  -79.154401,  208.58),
]

def gps_to_ecef_custom(lat, lon, alt):
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

def run_test():
    for pt in coords:
        print('custom', gps_to_ecef_custom(pt[0], pt[1], pt[2]))

if __name__ == "__main__":
    pass