import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# easiest way to load up data into numpy
test = np.loadtxt('testing.txt')

# split up the likelihoods and train data into that of birds and planes
likelihood = np.loadtxt('likelihood.txt')
# MAKE SURE THIS IS (400,) 1D ARRAY)----------------------------------------------------------------------------------------------------
bird_likelihood = likelihood[0]
plane_likelihood = likelihood[1]

data = np.loadtxt("dataset.txt")
bird_train = data[:10]
plane_train = data[10:]



#SHAPES ALREADY VERIFIED---------------------------------------------------------------------------------------------------------------



# set the priors
bird_prior = 0.5
plane_prior = 0.5

# we can actually just use fourier transforms to fit a function to the data. This way I dont need to round the data, to hopefully get a
# better estimate (even though I doubt it matters)




# desconstruct pdf to make a bird_likelihood fourier continuous function-------------------------------------------------------------------------
class FourierFunction:
    def __init__(self, likelihood_func):
        self.x = np.arange(len(likelihood_func))

        # decompose datapoints into set of sin waves
        fft_values = fft(likelihood_func)
        frequencies = fftfreq(len(likelihood_func), d=1)
        num_frequencies = 1000
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

    # use the decomposed sin waves to reconstruct the signal, effectively make a smooth function over all the datapoints
    def probability(self, query):
        result = 0
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        result = max(result, 0)

        total_area = sum(
            max(sum(amplitude * np.cos(2 * np.pi * frequency * i + phase)
                    for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)), 0)
            for i in self.x
        )

        # Return the normalized value
        return result / total_area

bird_func = FourierFunction(bird_likelihood)

print(bird_func.probability(200))

# # Test the Fourier function
# query_speed = 100  # Replace with the desired speed
# result = normalized_fourier_function(query_speed)
# print(f"Probability at speed {query_speed} km/h: {result:.4f}")


# # Test the Fourier function
# query_speed = 100  # Replace with the desired speed
# result = normalized_fourier_function(query_speed)
# print(f"Probability at speed {query_speed} km/h: {result:.4f}")

# # Plot the function over a range of speeds
# import matplotlib.pyplot as plt
# speeds = np.linspace(0, len(y), 1000)  # Generate 1000 points between 0 and the maximum index
# values = [normalized_fourier_function(s) for s in speeds]

# plt.figure(figsize=(12, 6))
# plt.plot(speeds, values, label="Fourier Function", color="green")
# plt.title("Fourier Function as a Probability Distribution")
# plt.xlabel("Speed")
# plt.ylabel("Probability")
# plt.grid(True)
# plt.legend()
# plt.show()


# # representation of the pdf of the bird speeds
# plt.figure(figsize=(10, 6)) 
# plt.plot(bird_likelihood, marker="o", linestyle="", markersize=3, label="Data Points")
# plt.title("Visualization of (1, 400) Array")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()
