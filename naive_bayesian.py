import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# TAKE THE LIKELIHOOD FUNCTIONS AND FIT A FOURNIER FUNCTION OVER THEM TO GRAB PROBABILITY OF FLOAT VALUES------------------------------------
likelihood = np.loadtxt('likelihood.txt')
# MAKE SURE THIS IS (400,) 1D ARRAY)d
bird_likelihood_func = likelihood[0]
plane_likelihood_func = likelihood[1]

# desconstruct pdf to make a bird_likelihood fourier continuous function----------------------------------------------------------------------
class FourierFunction:
    def __init__(self, data):
        self.x = np.arange(len(data))
        self.data = data

        # Decompose data into Fourier components
        fft_values = fft(data)
        frequencies = fftfreq(len(data), d=1)  # Assuming time step of 1 second
        num_frequencies = 1000  # Limit to the most significant frequencies
        magnitude = np.abs(fft_values)
        sorted_indices = np.argsort(magnitude)[::-1]
        dominant_indices = sorted_indices[:num_frequencies]

        self.amplitudes = []
        self.phases = []
        self.frequencies_selected = []
        for idx in dominant_indices:
            self.amplitudes.append(np.abs(fft_values[idx]))
            self.phases.append(np.angle(fft_values[idx]))
            self.frequencies_selected.append(frequencies[idx])

        # Total area for normalization
        self.total_area = sum(max(sum(amplitude * np.cos(2 * np.pi * frequency * i + phase) 
            for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)),0) 
            for i in self.x)

    def velocity(self, query):
        """Reconstruct velocity using Fourier components."""
        result = 0
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        result = max(result, 0)  # Ensure non-negative velocity
        return result

    def acceleration(self, query):
        """Compute acceleration as the time derivative of velocity."""
        result = 0
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += -2 * np.pi * frequency * amplitude * np.sin(2 * np.pi * frequency * query + phase)
        return result

    def show_velocity(self):
        """Show the overlay of the original data and Fourier approximation."""
        fourier_vals = [self.velocity(x) for x in self.x]
        fourier_vals = (fourier_vals - np.min(fourier_vals)) / (np.sum(fourier_vals))

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.data, label="datapoints", marker='o', linestyle="none")
        plt.plot(self.x, fourier_vals, label="fourier approximation", linestyle="-")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Fourier Approximation Overlay")
        plt.legend()
        plt.show()

    def show_accel(self):
        """Show the overlay of the original data and Fourier approximation."""
        fourier_vals = [self.acceleration(x) for x in self.x]
        fourier_vals = (fourier_vals - np.min(fourier_vals)) / (np.sum(fourier_vals))

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, fourier_vals, label="fourier approximation", linestyle="-")
        plt.xlabel("measurements")
        plt.ylabel("Value")
        plt.title("fourier approximation of acceleration from velocities")
        plt.legend()
        plt.show()

# setting up the likelyhood probability functions
bird_velo_likelihood = FourierFunction(bird_likelihood_func)
plane_velo_likelihood = FourierFunction(plane_likelihood_func)
bird_velo_likelihood.show_velocity()
plane_velo_likelihood.show_accel()




# plotting fourier function vs original data ----------------------------------------------------------------------------------------------

# x_vals = np.arange(len(bird_likelihood))
# fourier_vals = [bird_func.prob(x) for x in x_vals]

# # plot the transformation between the data and fitted function
# plt.figure(figsize=(10, 6))
# plt.plot(x_vals, bird_likelihood, label="bird data", marker='o', linestyle="none")
# plt.plot(x_vals, fourier_vals, label="fourier function", linestyle="-")
# plt.xlabel("speed")
# plt.ylabel("likelihood")
# plt.title("bird fourier approx laid over original function")
# plt.legend()
# plt.show()

# x_vals = np.arange(len(plane_likelihood_func))
# fourier_vals = [plane_func.prob(x) for x in x_vals]

# plt.figure(figsize=(10, 6))
# plt.plot(x_vals, plane_likelihood_func, label="plane data", marker='o', linestyle="none")
# plt.plot(x_vals, fourier_vals, label="fourier function", linestyle="-")
# plt.xlabel("speed")
# plt.ylabel("likelihood")
# plt.title("plane fourier approx laid over original function")
# plt.legend()
# plt.show()