# The Kalman Filter is used with the following set of equations and the they are used step-by-step.

"""
Step 1: Prediction Step
Here we extrapolate the state and uncertainity

1) x_n+1,n = Fx_n,n + Gu_n

2)P_n+1,n = F. P_n,n .F^T + Q

Step 2: Update Step

Finding the Kalman Gain

3) K_n,n = P_n,n-1. H^T .(HP_n,n-1H^T + R_n)^-1

Using the Kalman Gain in updating the measurement

4) x_n,n = x_n,n-1 + K_n(z_n-Hx_n,n-1)

Similarly updating the uncertainity

5) P_n,n = (I - K_nH)P_n,n-1(I - K_nH)^T + K_nR_nK_n^T

Where Q is the process noise covraince, G is the input control matrix, H is the measurement matrix, R is the measurement noise covraince matrix
"""

import numpy as np
import matplotlib.pyplot as plt

# This function is used to load the provided data as per the given format: t,u,z
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:,0]
    u = data[:,1:4]
    z = data[:,4:7]
    return t,u,z

# This function executes the Kalman Filter
def kalman_filter(t, u, z, R, Q, data_type):
    # Defining the mass and size of the matrix as per the total timestamps
    mass = 0.027
    n = len(t)   

    # Initialising the A and B matrix as calculated in the TASK 1 
    A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

    B = np.array([[0,0,0],
              [0, 0, 0],
              [0, 0,0],
              [1 / mass, 0, 0],
              [0, 1 / mass, 0],
              [0, 0, 1 / mass]])
    
    # Initialising the Uncertainity Matrix (P) given as follows: P_n+1,n = F.P_n,n .F^T + Q where Q is the Process Noise Covariance     
    P = np.zeros((n,6,6))
    P[0] = np.diag([0.01,0.01,0.01,0.05,0.05,0.05])
    x_hat = np.zeros((n,6))

    # The Measurement Matri (H) varies for position and velocity hence using the if condition to define both conditions and respective matrix
    if data_type == 'position':
        x_hat[0] = np.concatenate([z[0], np.zeros(3)])
        H = np.array([[1, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0], 
                      [0, 0, 1, 0,0, 0]])
        
    if data_type == 'velocity':
        x_hat[0] = np.concatenate([np.zeros(3),z[0]])
        H = np.array([[0, 0, 0, 1, 0, 0], 
                      [0, 1, 0, 0, 1, 0], 
                      [0, 0, 1, 0,0, 1]])
        
    # Executing the FOR loop over the entire timestamp
    for i in range(1,len(t)):

        #The delta time is the current - pervious time
        dt = t[i] - t[i-1]

        #Initialising x_n, p_n, u_n, z_n as previous values
        x_n = x_hat[i-1]
        u_n = u[i-1]
        P_n = P[i-1]
        z_n = z[i]
        # The F and G matrices convert the above defined A and B matrices (in the continuous domain) to the discrete time domain
        # for each time step.
        F = np.eye(6) + A*dt
        G = np.dot(np.eye(6) * dt + A * (dt ** 2), B)

        #This is the first step: Predict step
        x_predict = np.dot(F,x_n) + np.dot(G,u_n)

        P_predict = np.dot(np.dot(F,P_n),F.T) + Q

        # This is the second step the Update step
        y = z_n - (np.dot(H,x_predict)) 

        K = np.dot(np.dot(P_predict,H.T),np.linalg.inv(np.dot(np.dot(H,P_predict),H.T)+R))
        
        # Finally returning the states
        x_hat[i] = x_predict + K@y
        P[i] = np.dot((np.eye(6) - np.dot(K,H)), P_predict)
    
    return x_hat, P

# This is the main function which is responsbile to run the code where we define the Q and R values which vary for different file types
def main(data_type,file):
    
    # Different Q and R values for different file types
    if file == "kalman_filter_data_low_noise.txt":
        Q = np.eye(6)*((0.01)**2)
        R = np.eye(3)*((0.4)**2)       
    
    elif file == "kalman_filter_data_mocap.txt":
        Q = np.eye(6)*((0.001)**2)
        R = np.eye(3)*((0.01)**2)

    elif file == "kalman_filter_data_high_noise.txt":
        Q = np.eye(6)*((0.003)**2)
        R = np.eye(3)*((0.9)**2)

    elif file == "kalman_filter_data_velocity.txt":
        Q = np.eye(6)*((0.01)**2)
        R = np.eye(3)*((0.2)**2)

    # Loading the data and calling the kalman filter function to plot
    t, u, z = load_data(file)
    x_hat, P = kalman_filter(t, u, z, R, Q, data_type)

    P_plot = x_hat[:, :3]
    
    # Plotting the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P_plot[:, 0], P_plot[:, 1], P_plot[:, 2])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.title('3D Position Tracking with Kalman Filter')
    plt.show()


main(data_type='position', file='kalman_filter_data_mocap.txt')
main(data_type='position', file='kalman_filter_data_low_noise.txt')
main(data_type='position', file='kalman_filter_data_high_noise.txt')
main(data_type='velocity', file='kalman_filter_data_velocity.txt')