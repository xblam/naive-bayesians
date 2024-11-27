import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
''' 
THOUGHT PROCESS:
* already have the datapoints for the pdf of velocity, so just need to fit the fourier curve to the data
* use the dataset.txt of values to create a pdf for acceleration, which can be done just by taking the differential, of the velocity recordings
    - do not need to do this for velocity since we are already given the pdf
    - also make sure to get rid of Nan values
* fit the fourier curve to the acceleration
* now run naive bayesian taking into account both the velocity and accelereation likelihoods
* profit
'''

# TAKE THE LIKELIHOOD FUNCTIONS AND FIT A FOURNIER FUNCTION OVER THEM TO GRAB PROBABILITY OF FLOAT VALUES------------------------------------
likelihood = np.loadtxt('likelihood.txt')
# MAKE SURE THIS IS (400,) 1D ARRAY)d
bird_likelihood_func = likelihood[0]
plane_likelihood_func = likelihood[1]

# takes a set of pdf datapoints and makes them into a smooth pdf function
class FourierFunction:
    def __init__(self, data):
        self.x = np.arange(len(data))
        self.data = data

    def fournier_analysis(self)

        # Decompose data into Fourier components
        fft_values = fft(self.data)
        frequencies = fftfreq(len(self.data), d=1)  # in our data the step time is 1 second. Change if needed
        num_frequencies = 1000  # change if want more fidelity but 1000 should be fine
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

        # use total area to normalize this
        self.total_area = sum(max(sum(amplitude * np.cos(2 * np.pi * frequency * i + phase) 
            for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)),0) 
            for i in self.x)

        result = 0

        # we are not ensuring non-negativity, but should be fine
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        
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