import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import Counter

'''
* process the velocity pdf

* using the velocity train data, make a dict of acceleration pdfs
    - clean up the velocities by row filling
    - differentiate the velocities
    - pad the end of beginning with 0
    - sort the dictionary
    - make it into a valid pdf. If we get values that are not keys of the dict, we just return 0 for probs

* running actual baysian
    - first need to read in velocity data
    - clean up velocities data by row filling
    - get the acceleration data
    - now with these two sets of data run the naive baysian

* should i run this with numpy or pandas?
'''

# GET VELOCITY PDF---------------------------------------------------------------------------------------------------------------
velocity_pdf = np.loadtxt('likelihood.txt')

bird_velocity_list = velocity_pdf[0]
plane_velocity_list = velocity_pdf[1]

plt.plot(plane_velocity_list)
# plt.show()

# GET ACCELERATION PDF------------------------------------------------------------------------------------------------------------------------
train_data = np.loadtxt('dataset.txt')

# we need a function to row fill the velocity data
def row_fill(data):
    for i in range(data.shape[0]):
        row = data[i]
        valid = ~np.isnan(row)
        if np.any(valid):
            data[i] = np.interp(np.arange(len(row)), np.where(valid)[0], row[valid])
    return data

# fill in the nan values
train_data = row_fill(train_data)
train_data = train_data.round(0) # round the data to integers

# split into bird and plane
bird_train_data = train_data[:10].flatten()
plane_train_data = train_data[10:].flatten()

# get accel values, turn them all postive and make a pdf out of them
def get_acceleration_pdf(velocity_data):
    accel_data = np.diff(velocity_data)
    accel_data = np.abs(accel_data)
    num_datapoints = len(accel_data)
    accel_pdf = dict(sorted(Counter(accel_data).items()))
    standardized_pdf = divided_dict = {key: value / num_datapoints for key, value in accel_pdf.items()}
    return standardized_pdf

# get accel values (they are all postive)
bird_acceleration_dict = get_acceleration_pdf(bird_train_data)
plane_acceleration_dict = get_acceleration_pdf(plane_train_data)

# UNABLE TO DIFFERENTIATE BETWEEN ACCELERATION AND VELOCITY WITH ACCELERATION AND VELOCITY ---------------------------------------------------
# USE FOURIER SERIES TO BOTH BUILD SMOOTHER PDFS, AND TRACK OCCILATIONS SINCE BIRDS OCCILATE SPEEDS MORE


# FOURIER ANALYSIS ON THE GIVEN DATA POINTS. RETURNS SMOOTH FUNCTION -------------------------------------------------------------------
class FourierFunction:
    def __init__(self, data):
        self.x = np.arange(len(data))
        self.data = data

        # Decompose data into Fourier components
        fft_values = fft(self.data)
        frequencies = fftfreq(len(self.data), d=1)  # in our data the step time is 1 second. Change if needed
        num_frequencies = 1000  # change if want more fidelity but 1000 should be fine
        magnitude = np.abs(fft_values)
        sorted_indices = np.argsort(magnitude)[::-1]
        dominant_indices = sorted_indices[:num_frequencies]

        # set up values to reconstruct the function
        self.amplitudes = []
        self.phases = []
        self.frequencies_selected = []
        for idx in dominant_indices:
            self.amplitudes.append(np.abs(fft_values[idx]))
            self.phases.append(np.angle(fft_values[idx]))
            self.frequencies_selected.append(frequencies[idx])

        # use to normalize since we are dealing with probabilities
        self.total_area = sum(max(sum(amplitude * np.cos(2 * np.pi * frequency * i + phase) 
            for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)),0) 
            for i in self.x)
    
    # call this property to get the approximation at input value
    def fourier(self, query):
        result = 0

        # we are not ensuring non-negativity, but should be fine
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        
        return result/self.total_area

    # overlay approximation function over original data points for sanity check
    def show_function(self):
        """Show the overlay of the original data and Fourier approximation."""
        fourier_vals = [self.fourier(x) for x in self.x]
        fourier_vals = (fourier_vals - np.min(fourier_vals)) / (np.sum(fourier_vals))

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.data, label="datapoints", marker='o', linestyle="none")
        plt.plot(self.x, fourier_vals, label="fourier approximation", linestyle="-")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Fourier Approximation Overlay")
        plt.legend()
        plt.show()

    # this is to make it so the instance can be called as a function
    def __call__(self, query):
        return self.fourier(query)

# NOW PUT IT ALL TOGETHER IN NAIVE BAYESIAN MODEL---------------------------------------------------------------------------------------


def naive_bayesian(velocity, acceleration):
    classifications = []
    bird_posterior = 0.5
    plane_posterior = 0.5

    for vel, acc in zip(velocity, acceleration):

        # it turns out that the classification is more accurate if we disregard velocity, but it is a feature that could still be implemented.
        vel = int(vel)

        # Check for valid range in PDFs
        bird_velocity_likelihood = bird_velocity_list[vel]
        plane_velocity_likelihood = plane_velocity_list[vel]
        bird_acceleration_likelihood = bird_acceleration_dict.get(acc, 1e-8)
        plane_acceleration_likelihood = plane_acceleration_dict.get(acc, 1e-8)

        # Calculate likelihoods
        bird_likelihood = bird_acceleration_likelihood
        plane_likelihood = plane_acceleration_likelihood

        # update posteriors using transition probabilities
        bird_posterior = bird_likelihood * (bird_posterior * 0.9 + plane_posterior * 0.1)
        plane_posterior = plane_likelihood * (plane_posterior * 0.9 + bird_posterior * 0.1)

        # we have to normalize
        normalization_factor = bird_posterior + plane_posterior
        if normalization_factor > 0:
            bird_posterior /= normalization_factor
            plane_posterior /= normalization_factor


        # then just classify
        classifications.append('b' if bird_posterior > plane_posterior else 'a')

    # and return majority vote
    track_class = 'b' if classifications.count('b') > classifications.count('a') else 'a'
    return classifications, track_class

# NOW BUILD LOOP FOR CLASSIFICATION---------------------------------------------------------------------------------------------------------

test_data = np.loadtxt('testing.txt')
test_data = row_fill(test_data)
test_data = test_data.round(0)


for i, velocity in enumerate(test_data):
    #  load velocity, and mkae sure the the last thing of acceleration is 0
    acceleration = np.diff(velocity)# might have to add one value to the end to match 600?
    acceleration = np.append(acceleration, 0)
    acceleration = np.abs(acceleration)
    sample_classifications, track_classification = naive_bayesian(velocity, acceleration)
    print(f'object {i+1}, class {track_classification}')


