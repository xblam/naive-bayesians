import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# TAKE THE LIKELIHOOD FUNCTIONS AND FIT A FOURNIER FUNCTION OVER THEM TO GRAB PROBABILITY OF FLOAT VALUES------------------------------------
likelihood = np.loadtxt('likelihood.txt')
# MAKE SURE THIS IS (400,) 1D ARRAY)d
bird_likelihood_func = likelihood[0]
plane_likelihood_func = likelihood[1]

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

        self.total_area = sum(
            max(sum(amplitude * np.cos(2 * np.pi * frequency * i + phase)
                    for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)), 0)
            for i in self.x
        )

    # use the decomposed sin waves to reconstruct the signal, effectively make a smooth function over all the datapoints
    def prob(self, query):
        result = 0
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        result = max(result, 0)

        # Return the normalized value
        return result / self.total_area

# setting up the likelyhood probability functions
bird_func = FourierFunction(bird_likelihood_func)
plane_func = FourierFunction(plane_likelihood_func) 

# preprocessing for train data--------------------------------------------------------------------------------------------------------
train_data = np.loadtxt("dataset.txt")

bird_train = train_data[:10]
plane_train = train_data[10:]

# replace Nan with mean of each track
bird_train = np.nan_to_num(bird_train, nan=np.nanmean(bird_train))
plane_train = np.nan_to_num(plane_train, nan=np.nanmean(plane_train))

# set the priors
bird_prior = 0.5
plane_prior = 0.5

# preprocessing for test data--------------------------------------------------------------------------------------------------------------------------------------
test_data = np.loadtxt("testing.txt")

def classify_velocity_with_transition(velocity, prev_class=None, transition_prob=0.7):
    # Calculate likelihoods for bird and plane
    bird_likelihood = bird_func.prob(velocity)  # P(velocity | bird)
    plane_likelihood = plane_func.prob(velocity)  # P(velocity | plane)

    # Calculate initial posteriors
    bird_posterior = bird_likelihood * 0.5  # P(bird) = 0.5
    plane_posterior = plane_likelihood * 0.5  # P(plane) = 0.5

    # Adjust posteriors based on transition probabilities
    if prev_class is not None:
        if prev_class == "bird":
            bird_posterior *= transition_prob  # Favor staying in "bird"
            plane_posterior *= (1 - transition_prob)  # Penalize switching to "plane"
        elif prev_class == "plane":
            plane_posterior *= transition_prob  # Favor staying in "plane"
            bird_posterior *= (1 - transition_prob)  # Penalize switching to "bird"

    # Normalize posteriors
    total_posterior = bird_posterior + plane_posterior
    bird_posterior /= total_posterior
    plane_posterior /= total_posterior

    # Classify based on higher posterior
    return ("bird", bird_posterior) if bird_posterior > plane_posterior else ("plane", plane_posterior)

def classify_object(velocities, transition_prob=0.7):
    classifications = []
    prev_class = None

    # Classify each velocity
    for velocity in velocities:
        current_class, _ = classify_velocity_with_transition(velocity, prev_class, transition_prob)
        classifications.append(current_class)
        prev_class = current_class  # Update for the next velocity

    # Final classification: majority vote
    final_class = "bird" if classifications.count("bird") > classifications.count("plane") else "plane"
    return classifications, final_class

# Example input: velocities for one object (600 measurements)
object_velocities = [310, 420, 250, 620, 500, ...]  # Replace with actual velocity data

# Classify the object
for i in range(10):
    sample_classes, final_class = classify_object(test_data[i])

        # Print results
    print(f"Final class for the object: {final_class}")



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