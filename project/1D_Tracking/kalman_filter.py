import sys
import numpy as np

class KalmanFilter1d(object):
    def __init__(self, params):

        # Get the parameters for Kalman filter 
        dt=params["dt"]
        u=params["u"]        
        std_meas = params["std_meas"]         
        std_acc = params["std_acc"]
        
        self.x = np.array([[0], 
                           [0]])        # 2x1 array
        self.u = u                
        
        self.A = np.array([[1, dt], 
                           [0, 1]])     # 2x2 array
        
        self.B = np.array([[(dt**2)/2], 
                           [dt]])       # 2x1 array 
        
        self.H = np.array([[1, 0]])     # 1x2 array

        self.Q = np.array([[(dt**4)/4, (dt**3)/2],
                            [(dt**3)/2, dt**2]]) * std_acc**2   # 2x2 array
                        
        self.R = std_meas**2                                
        
        self.P = np.eye(self.A.shape[1]) * 0.0001    # 2x2 array

    def predict(self):
        # Predict state (Eq 2.4)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        # Calculate error covariance (Eq 2.5)
        # P= A*P*A' + Q        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q        
        if np.isnan(self.P).any():
            print("Unable to continue tracking: The parameters are not properly tuned.")
            sys.exit()
        return self.x

    def update(self, z):
        # Calculate the Kalman Gain (Eq 2.6)
        # K = P * H'* inv(H*P*H'+R)      
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        # update the state using the new measurement  (Eq 2.7)
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
            
        # update the error covariance matrix (Eq 2.8)
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K,self.H)),self.P)
        
        return self.x