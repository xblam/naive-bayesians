import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

speed_likelihoods = np.loadtxt('likelihood.txt')
dataset = np.loadtxt('dataset.txt')

class MakePDFs:
    def __init__(self, speed_likelihoods):
        self.speed_likelihoods = speed_likelihoods
        self.make_v_pdf()

    # make the velocity pdfs for birds and planes
    def make_v_pdf(self):
        self.bird_probs = self.speed_likelihoods[0]
        self.plane_probs = self.speed_likelihoods[1]

        self.bird_speed = np.arange(len(self.bird_probs))
        self.plane_speed = np.arange(len(self.plane_probs))

        self.bird_v_pdf = interp1d(self.bird_speed, self.bird_probs, kind='cubic', fill_value="extrapolate")
        self.plane_v_pdf = interp1d(self.plane_speed, self.plane_probs, kind='cubic', fill_value='extrapolate')



    def graph_speed(self):
        test_speeds = np.linspace(0, len(self.bird_probs)-1, 1000)
        interpolated_bird = self.bird_v_pdf(test_speeds)
        interpolated_plane = self.plane_v_pdf(test_speeds)

        plt.figure(figsize=(10, 6))
        plt.plot(self.bird_speed, self.bird_probs, 'o', label="original")
        plt.plot(test_speeds, interpolated_bird, '-', label="interpolated")
        plt.xlabel("Speed")
        plt.ylabel("Probability")
        plt.title("bird interpolated")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.plane_speed, self.plane_probs, 'o', label="original")
        plt.plot(test_speeds, interpolated_plane, '-', label="Interpolated")
        plt.xlabel("Speed")
        plt.ylabel("Probability")
        plt.title("plane interpolated")
        plt.legend()
        plt.show()


    def get_accel_values(self):
        bird_v = dataset[:10]
        plane_v = dataset[10:]
        print(bird_v.shape)
        print(plane_v.shape)
        acceleration_data = np.diff(velocity_data, axis=1)
ben = MakePDFs(speed_likelihoods)
ben.graph_speed()

# def graph_speed(func, probs, speed):
#    # Test the function on float inputs
#     test_speeds = np.linspace(0, 399, 1000)  # Fine-grained speed values
#     interpolated_probs = func(test_speeds)

#     # Plot original points and the interpolated curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(speed, probs, 'o', label="Original Points")
#     plt.plot(test_speeds, interpolated_probs, '-', label="Interpolated Function")
#     plt.xlabel("Speed")
#     plt.ylabel("Probability")
#     plt.title("Interpolated Function for Float Inputs")
#     plt.legend()
#     plt.show()

# # graph_speed(bird_speed_func, bird_probs, bird_speed)
# graph_speed(plane_speed_func, plane_probs, plane_speed)