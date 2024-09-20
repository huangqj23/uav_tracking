import numpy as np
from utils.utils import bbox_x1y1x2y2_cxcywh,bbox_cxcywh_x1y1x2y2
import copy

class KalmanBBoxMot(object):

    def __init__(self, params, bbox):                
        # Get the parameters for Kalman filter 
        dt=params["dt"]
        u_x=u_y=params["u"]  
        std_acc = params["std_acc"]      
        std_meas_x = params["std_meas_x"]  
        std_meas_y = params["std_meas_y"]                 
        
        # Define the  control input matrix with a size 2x1 array
        self.u = np.array([[u_x],
                           [u_y]])

        # Initialize the initial state with a size 6x1 array
        self.x = np.array([[0], 
                           [0], 
                           [0], 
                           [0],
                           [0],
                           [0]]) 
            
        # Assign initial state with detection bbox in format [c_x, c_y, w , h]
        self.x[:4] = bbox_x1y1x2y2_cxcywh(bbox)
    
        # State transition matrix A
        self.A = np.array([
                            [1, 0, 0, 0, dt, 0],  
                            [0, 1, 0, 0, 0, dt],
                            [0, 0, 1, 0,  0, 0],
                            [0, 0, 0, 1,  0, 0],
                            [0, 0, 0, 0,  1, 0],
                            [0, 0, 0, 0,  0, 1]])  # 6x6 array
        
        # Control Input Matrix B , array 6 x 2        
        self.B = np.array([
                            [(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [0,  0],
                            [0,  0],
                            [dt, 0],
                            [0, dt]])              # 6x2 array

        # Measurement Matrix H
        self.H = np.array([
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]])   # 4x6 array

        # Covariance noise matrix Q , an  6 x 6 array
        self.Q = np.array([[(dt**4)/4, 0, 0, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, 0, 0, (dt**3)/2],
                           [0, 0, 1e-5, 0, 0, 0],
                           [0, 0, 0, 1e-5, 0, 0],
                           [(dt**3)/2, 0, 0, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, 0, 0, dt**2]
                           ]) * std_acc**2         # 6x6 array

        # Initial Measurement Noise Covariance R 
        self.R = np.array([[std_meas_x**2, 0 , 0 , 0],
                           [0, std_meas_y**2, 0 , 0],
                           [0, 0, 1e-10, 0],
                           [0, 0, 0, 1e-10]])      # 4x4 array
                                
        #Initial Covariance Matrix P
        self.P = np.eye(self.A.shape[1]) * 1000

        # set initial  number of detections for a track
        self.detect_count = 1 
        #self.lost_count = 0 
        

    def predict(self):        
        #self.lost_count += 1
        """Time Update Equations"""
        # Calculate a priori state: x_k =Ax_(k-1) + Bu_(k-1)  
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)        
 
        # Calculate a priori error covariance matrix: P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        predicted_state = copy.deepcopy(self.x)
        
        # convert bbox to format [x1, y1, x2, y2]
        predicted_state[:4] = bbox_cxcywh_x1y1x2y2(predicted_state[:4])

        return predicted_state


    def update(self,z):
        self.lost_count = 0 
        """Measurement Update Equations"""
        # convert detection from format [x1, y1, x2, y2] to [c_x, c_y, w , h]
        z=bbox_x1y1x2y2_cxcywh(z)

        # Update the total number of detections for the same track ID
        self.detect_count += 1
        
        # Calculate the Kalman Gain: K = P * H'* inv(H*P*H'+R)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R    
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        # Calculate a posteriori state estimate          
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))              

        I = np.eye(self.H.shape[1])

        # Calculate a posteriori error covariance matrix        
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        updated_state = copy.deepcopy(self.x)

        # convert bbox to format [x1, y1, x2, y2]
        updated_state[:4] = bbox_cxcywh_x1y1x2y2(updated_state[:4])

        return updated_state