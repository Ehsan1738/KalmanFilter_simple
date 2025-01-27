"""
Ehsan Rahimi
"""

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dt=0.3, process_variance=1e-4, measurement_variance=0.0001):

        """
        Initializing the Kalman filter with the necessary parameters and matrices.

        Parameters:
        - dt: The time step between measurements. Default is 0.3 seconds.
        - process_variance: The variance of the process noise, which represents 
        the uncertainty in the system model. Default is 1e-4.
        - measurement_variance: The variance of the measurement noise, which 
        represents the uncertainty in the sensor measurements. Default is 0.0001.

        Attributes:
        - column (list): A placeholder for storing data (not used in this implementation).
        - dt (float): The time step between measurements.
        - x (numpy array): The initial state vector, representing the system's state (e.g., [position, velocity]).
        - A (numpy array): The state transition matrix, which predicts the next state based on the current state.
        - Q (numpy array): The process noise covariance matrix, which models the uncertainty in the system dynamics.
        - H (numpy array): The observation matrix, which maps the state vector to the measurement space.
        - R (numpy array): The measurement noise covariance matrix, which models the uncertainty in the sensor measurements.
        - P (numpy array): The initial error covariance matrix, which represents the uncertainty in the state estimate.
        """
        self.column = []
        self.dt = dt
        self.x = np.zeros((2, 1))  # Initial state: [pos, vel] (no acceleration)
        self.A = np.array([[1, self.dt],
                           [0, 1]])
        self.Q = np.array([[self.dt**4 / 4, self.dt**3 / 2],
                           [self.dt**3 / 2, self.dt**2]]) * process_variance

        self.H = np.array([[1, 0]])
        self.R = np.array([[measurement_variance]])
        self.P = np.eye(2)

    def predict(self):
        """
        This step performs the prediction step of the Kalman filter. 

        Steps:

        1. Predicts the next state (`x`) using the state transition matrix (`A`).
        2. Predicts the next error covariance matrix (`P`) using the state transition matrix (`A`) 
        and the process noise covariance matrix (`Q`).
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, measurement):
        """
        This function performs the update step of the Kalman filter, 
        which incorporates a new measurement to refine the current 
        state estimate and uncertainty matrix.

        Steps:
        1. z = np.array([[measurement]]): 
        - The measurement (position) is formatted as a 1x1 numpy array for matrix operations.

        2. y = z - np.dot(self.H, self.x): 
        - Compute the measurement residual (y), which is the difference between 
            the actual measurement (z) and the predicted measurement (H * x). 
            This shows how much the predicted state deviates from the observed state.

        3. S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R: 
        - Compute the residual covariance (S), which is a measure of uncertainty in the measurement.
        - It combines the projected state covariance (H * P * H^T) and the measurement noise covariance (R).

        4. K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)): 
        - Compute the Kalman gain (K), which determines how much weight to give to the measurement 
            versus the prediction when updating the state estimate.

        5. self.x = self.x + np.dot(K, y): 
        - Update the state estimate (x) by adding the correction term (K * y), 
            which adjusts the prediction based on the measurement residual scaled by the Kalman gain.

        6. I = np.eye(self.P.shape[0]): 
        - Create an identity matrix (I) of the same size as the state covariance matrix (P).

        7. self.P = np.dot((I - np.dot(K, self.H)), self.P): 
        - Update the state covariance matrix (P) by reducing uncertainty in directions 
            where the measurement has provided information.
        - The term (I - K * H) reduces the covariance in the directions improved by the measurement.

        The updated state (self.x) and covariance (self.P) are now refined.
        """
        z = np.array([[measurement]])  # Measurement is just the position
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def data(self):
        with open("Assignment_gps_filter.txt", "r") as file:
            for line in file.readlines():
                line1 = float(line.replace(",", ".").split("\t")[0])
                self.column.append(line1)
        return self.column

    def plot_graph(self, gps_data, estimated_positions):
        plt.figure(figsize=(10, 6))
        plt.plot(gps_data, label='Measured Position Data', color='blue', linewidth=1)
        plt.plot(estimated_positions, label='Filtered Position Data', color='red', linewidth=1)
        plt.title('Measured vs Filtered Position Data')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def main():
        """
        Main function demonstrates the Kalman Filter.

        Steps:
        1. Initializes a Kalman filter object (`kf`).
        2. Loads GPS data using the `data` method of the Kalman filter.
        3. Iterates through each GPS measurement:
            a. Predicts the next state using the Kalman filter's `predict` method.
            b. Updates the state estimate using the Kalman filter's `update` method and the current GPS measurement.
            c. Stores the estimated position in the `positions` list.
        4. Plots the original GPS data and the filtered positions using the `plot_graph` method.

        """
        kf = KalmanFilter()
        gps_data = kf.data()

        positions = []
        for pos in gps_data:
            kf.predict()
            kf.update(pos)
            positions.append(kf.x[0, 0])
            
        print(positions[-3:], gps_data[-3:])
        kf.plot_graph(gps_data, positions)



if __name__ == "__main__":
    KalmanFilter.main()