import numpy as np
from kalman_filter import KalmanFilter1d
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json
import argparse

def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

# Define the trajectory function
def trajectory(duration, step):
    # create time steps from 0 to the value specified by duration
    t = np.arange(0, duration, step)  
    # true trajectory function
    Y = 0.01*(t**2 - t)
    return Y,t

def main(args):

    # Define the true trajectory with duration = 50s, step =0.1s
    Y,t = trajectory(50, 0.1)
    
    # Load parameters from JSON file
    tracker_params = load_json(args.tracker_params)
    
    # Create KalmanFilter object
    KF = KalmanFilter1d(tracker_params)

    # Define lists for storing the prediction, measurement, and estimated positions 
    meas_pos = []
    pred_pos = []
    est_pos = []
    
    # Define the std of the normal distribution used to create the measurement noise
    noise_std = 1

    for y in Y:        
                                              
        predicted_state = KF.predict()        
        pred_pos.append(predicted_state[0]) 

        # add noise to simulate the measurement process        
        y = y  + np.random.normal(0, noise_std) 
        meas_pos.append(y)
        
        updated_state=KF.update(y)        
        est_pos.append(updated_state[0])
            
    #### PLOT THE RESULTS  ####
    fig = plt.figure()    
    fig.suptitle('Tracking Object in 1-D space with Kalman Filter\n \
    Positions (m) vs Times (s)', fontsize=15, weight='bold')

    plt.plot(t, meas_pos, label='Measurement', color='b',linewidth=0.5)
    plt.plot(t, est_pos, label='Estimated (KF)', color='red',linewidth=1)
    plt.plot(t, Y, label='True Trajectory', color='yellow', linewidth=1)
    #plt.plot(t, pred_pos, label='Predicted (KF)', color='k', linewidth=0.5)

    plt.xlabel('Times (s)', fontsize=15)
    plt.ylabel('Positions (m)', fontsize=15)
    plt.legend(fontsize=15)
        
    ### CALCULATE MEANS SQUARED ERRORS (MSE)   
    mse_est = mean_squared_error(Y,est_pos)        
    mse_meas = mean_squared_error(Y,meas_pos)
    print(f"Estimated (KF) MSE : {round(mse_est,3)} m")
    print(f"Measurement MSE: {round(mse_meas,3)} m")
    
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='1-D Kalman Filter')
    parser.add_argument('--tracker-params', type=str, \
                        default='tracker_params.json')
    args = parser.parse_args()      

    main(args)