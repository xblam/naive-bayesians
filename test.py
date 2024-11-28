import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import Counter
''' 
THOUGHT PROCESS:
* already have the datapoints for the pdf of velocity, so just need to fit the fourier curve to the data
* use the dataset.txt of values to create a pdf for acceleration, which can be done just by taking the differential of the velocity recordings
    - do not need to do this for velocity since we are already given the pdf
    - also make sure to get rid of Nan values
* fit the fourier curve to the acceleration pdf datapoints
* now run naive bayesian taking into account both the velocity and accelereation likelihoods
* profit
'''

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


# takes in datapoints of velocity and then returns datapoints of acceleration
def get_accel_data(dataset):
    accel_data = dataset.apply(lambda row: row.diff(), axis=1) # differentiate
    accel_data = accel_data.fillna(0) # fill na with 0
    return accel_data

# given a set of train velocities, get the accel pdf datapoints
def get_accel_helper(dataset):
    a_vals = get_accel_data(dataset) # take acceleration vals from velocity vals
    a_data = a_vals.values.flatten()
    a_pdf_data = (Counter(a_data).values()) # count to make imperical pdf
    total = sum(a_pdf_data)
    a_pdf = [value / total for value in a_pdf_data]
    return a_pdf


def get_accel_pdf(dataset):
    bird_v_vals = dataset[:10]
    plane_v_vals = dataset[10:]
    bird_a_pdf = get_accel_helper(bird_v_vals) 
    plane_a_pdf = get_accel_helper(plane_v_vals)
    return bird_a_pdf, plane_a_pdf


def naive_recursive_bayesian_classifier(v, a, bird_v_pdf, plane_v_pdf, bird_a_pdf, plane_a_pdf, transition_prob=0.9):
    """
    Classify data points using velocity and acceleration likelihoods with recursive Bayesian updating.
    """
    classifications = []
    posteriors = []

    P_bird = 0.5
    P_plane = 0.5

    for v, a in zip(velocity, acceleration):
        # Likelihoods
        P_v_given_bird = bird_v_pdf.pdf(v)
        P_a_given_bird = bird_a_pdf.pdf(a)
        P_v_given_plane = plane_v_pdf.pdf(v)
        P_a_given_plane = plane_a_pdf.pdf(a)

        # Posteriors
        P_bird_given_data = P_v_given_bird * P_a_given_bird * P_bird
        P_plane_given_data = P_v_given_plane * P_a_given_plane * P_plane

        # Normalize
        total = P_bird_given_data + P_plane_given_data
        P_bird_given_data /= total
        P_plane_given_data /= total

        # Update with transition probability
        P_bird = transition_prob * P_bird_given_data + (1-transition_prob) * P_plane_given_data
        P_plane = transition_prob * P_plane_given_data + (1-transition_prob) * P_bird_given_data

        # Classify
        classification = 'b' if P_bird > P_plane else 'a'
        classifications.append(classification)
        posteriors.append((P_bird, P_plane))

    return classifications, posteriors







if __name__ == '__main__':
    # MAKE VELOCITY PDFS------------------------------------------------------------------------------------------------
    likelihood = pd.DataFrame(np.loadtxt('likelihood.txt'))
    bird_v_pdf = likelihood[:1].values.flatten()
    plane_v_pdf = likelihood[1:].values.flatten()

    # USING TRAINING VELOCITY DATA MAKE ACCELERATION PDF----------------------------------------------------------------
    dataset = pd.DataFrame(np.loadtxt('dataset.txt')).round(0)
    dataset = dataset.fillna(method='ffill', axis=1)
    dataset = dataset.fillna(0)
    data_test = dataset[10:].values.flatten()

    bird_a_pdf, plane_a_pdf = get_accel_pdf(dataset)


    # PAST THIS POINT WE HAVE BOTH PDFS--------------------------------------------------------------------------------

    results = []
    for velocity, acceleration in zip(testing_velocity, testing_acceleration):
        classifications, posteriors = naive_recursive_bayesian_classifier(
            velocity, acceleration,
            bird_velocity_likelihood, plane_velocity_likelihood,
            bird_acceleration_likelihood, plane_acceleration_likelihood
        )
        results.append({
            "classifications": classifications,
            "track_summary": max(set(classifications), key=classifications.count),  # Majority class
            "posteriors": posteriors
        })