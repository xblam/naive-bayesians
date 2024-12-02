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

print(sum(bird_velocity_list))
print(sum(plane_velocity_list))


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
    # print('pdf\n',standardized_pdf)
    return standardized_pdf

# get accel values (they are all postive)
bird_acceleration_dict = get_acceleration_pdf(bird_train_data)
plane_acceleration_dict = get_acceleration_pdf(plane_train_data)


# NOW PUT IT ALL TOGETHER IN NAIVE BAYESIAN MODEL---------------------------------------------------------------------------------------


def naive_bayesian(velocity, acceleration):
    classifications = []
    bird_posterior = 0.5
    plane_posterior = 0.5
    max_vel = int(max(velocity))
    print(type(max_vel))
    print(velocity[max_vel])

    for vel, acc in zip(velocity, acceleration):
        # Safely handle vel and acc
        vel = int(vel)

        # Check for valid range in PDFs
        bird_velocity_likelihood = bird_velocity_list[vel]
        plane_velocity_likelihood = plane_velocity_list[vel]
        bird_acceleration_likelihood = bird_acceleration_dict.get(acc, 1e-8)
        plane_acceleration_likelihood = plane_acceleration_dict.get(acc, 1e-8)

        # Calculate likelihoods
        bird_likelihood = bird_acceleration_likelihood + bird_velocity_likelihood
        plane_likelihood = plane_acceleration_likelihood + plane_velocity_likelihood
        # print('vel', vel)
        # print('acc', acc)
        # print('bird vel', bird_velocity_likelihood)
        # print('plane vel', plane_velocity_likelihood)
        # print('bird acc', bird_acceleration_likelihood)
        # print('plane acc', plane_acceleration_likelihood)
        # print(f'bird probs {bird_likelihood}')
        # print(f'plane probs {plane_likelihood}')

        # Update posteriors using transition probabilities
        bird_posterior = bird_likelihood * (bird_posterior * 0.9 + plane_posterior * 0.1)
        plane_posterior = plane_likelihood * (plane_posterior * 0.9 + bird_posterior * 0.1)

        # we have to normalize or else the values round down to 0 for some reason
        normalization_factor = bird_posterior + plane_posterior
        if normalization_factor > 0:
            bird_posterior /= normalization_factor
            plane_posterior /= normalization_factor
        # print(f'bird pos {bird_posterior}')
        # print(f'plane pos {plane_posterior}')

        # Classify this sample
        classifications.append('b' if bird_posterior > plane_posterior else 'a')

    # Final classification based on majority vote
    track_class = 'b' if classifications.count('b') > classifications.count('a') else 'a'
    print(track_class)
    
    return classifications, track_class

# NOW BUILD LOOP FOR CLASSIFICATION---------------------------------------------------------------------------------------------------------

test_data = np.loadtxt('testing.txt')
test_data = row_fill(test_data)
test_data = test_data.round(0)

# for i in range(400):
#     print((i))


velocity = test_data[8]
print(max(velocity))

for velocity in test_data:
    #  load velocity, and mkae sure the the last thing of acceleration is 0
    acceleration = np.diff(velocity)# might have to add one value to the end to match 600?
    acceleration = np.append(acceleration, 0)
    acceleration = np.abs(acceleration)
    # print(acceleration)
    sample_classifications, track_classification = naive_bayesian(velocity, acceleration)
    # print("Sample classifications:", sample_classifications)


