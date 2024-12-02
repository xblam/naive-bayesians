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
    accel_data = dataset.apply(lambda row: row.diff(), axis=1).round(0) # differentiate, round integers
    accel_data = accel_data.fillna(0) # fill na with 0
    return accel_data

# given a set of train velocities, get the accel pdf datapoints
def get_accel_helper(dataset):
    a_vals = get_accel_data(dataset) # take acceleration vals from velocity vals
    a_data = a_vals.values.flatten()

    a_data = dict(sorted(Counter(a_data).items()))
    print('a_data\n', a_data)
    print(type(a_data))
    a_pdf_data = list(a_data) # count to make imperical pdf
    print('pdf_data\n,', a_pdf_data)
    print('the sorted indexes of counter data \n', sorted(Counter(a_data)))
    print('a pdf data \n', a_pdf_data)
    total = sum(a_pdf_data)
    a_pdf_data = [value / total for value in a_pdf_data]
    return a_pdf_data


def get_accel_pdf(dataset):
    dataset = pd.DataFrame(np.loadtxt('dataset.txt'))
    bird_v_vals = dataset[:10]
    plane_v_vals = dataset[10:] 
    bird_pdf_data = get_accel_helper(bird_v_vals) 
    plane_pdf_data = get_accel_helper(plane_v_vals)
    bird_pdf = FourierFunction(bird_pdf_data)
    plane_pdf = FourierFunction(plane_pdf_data)
    return bird_pdf, plane_pdf


if __name__ == '__main__':
    # MAKE VELOCITY PDFS------------------------------------------------------------------------------------------------
    likelihood = np.loadtxt('likelihood.txt')
    bird_v_data = likelihood[0]
    plane_v_data = likelihood[1]
    bird_v_pdf = FourierFunction(bird_v_data)
    plane_v_pdf = FourierFunction(plane_v_data)

    # USING TRAINING VELOCITY DATA MAKE ACCELERATION PDF----------------------------------------------------------------
    dataset = pd.DataFrame(np.loadtxt('dataset.txt'))
    dataset = dataset.fillna(method='ffill', axis=1)
    dataset = dataset.fillna(0)
    bird_a_pdf, plane_a_pdf = get_accel_pdf(dataset)

    # PAST THIS POINT WE HAVE THE VELOCITY PDF AND ACCELERATION PDF-----------------------------------------------------
    # test_dataset = pd.DataFrame(np.loadtxt('testing.txt'))
    # print(test_dataset)
    # test_dataset = test_dataset.fillna(method='ffill', axis=1)
    # test_dataset = test_dataset.fillna(0)
    # print(test_dataset)
    # test_dataset = test_dataset.round(0)
    # print(test_dataset)

    '''
    bascially hte problem right now is that when I get the pdfs for accel, they do not line up properly in the sense that it is hard
    for me ot grpah accel to likleihood.

    also need to pad it so that if the range is outside, then prob of accel for plane will be 0

    okay so I have ti so that I have sorted the dict by the acceleration, and we havea number of occurences for each accel. We can normalize this
    to get a probaiblity, and then just use rounding to get the probability of the observation being of the object.
    
    '''
    