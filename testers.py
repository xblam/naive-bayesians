import numpy as np
from scipy.fftpack import fft, fftfreq

# Load likelihood functions
likelihood = np.loadtxt('likelihood.txt')
bird_likelihood_func = likelihood[0]
plane_likelihood_func = likelihood[1]

# Define the FourierFunction class
class FourierFunction:
    def __init__(self, data):
        self.x = np.arange(len(data))

        # Decompose data into Fourier components
        fft_values = fft(data)
        frequencies = fftfreq(len(data), d=1)  # Assuming time step of 1 second
        num_frequencies = 100  # Limit to the most significant frequencies
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
        self.total_area = sum(
            max(
                sum(amplitude * np.cos(2 * np.pi * frequency * i + phase)
                    for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases)), 0
            )
            for i in self.x
        )

    def prob(self, query):
        """Reconstruct probability using Fourier components."""
        result = 0
        for amplitude, frequency, phase in zip(self.amplitudes, self.frequencies_selected, self.phases):
            result += amplitude * np.cos(2 * np.pi * frequency * query + phase)
        result = max(result, 0)  # Ensure non-negative probability
        return result / self.total_area

# Instantiate Fourier functions for bird and plane likelihoods
bird_func = FourierFunction(bird_likelihood_func)
plane_func = FourierFunction(plane_likelihood_func)

# Preprocessing for train data
train_data = np.loadtxt("testing.txt")

def replace_nan_with_mean(data):
    """Replace NaN values in a 2D array with the row-wise mean."""
    for i in range(data.shape[0]):  # Loop through each row (track)
        row_mean = np.nanmean(data[i])  # Compute mean ignoring NaN
        data[i] = np.where(np.isnan(data[i]), row_mean, data[i])  # Replace NaN with row mean
    return data

def velo_and_accel(data):
    """Compute corrected velocity and acceleration for tracks."""
    velo = replace_nan_with_mean(data)
    accel = np.diff(velo, axis=1, prepend=np.nan)  # Compute acceleration
    return velo, accel

# Preprocess train data
velo_train, accel_train = velo_and_accel(train_data)

# Define Bayesian classification functions
def classify_point(velocity, acceleration, prev_class=None, transition_probs=None):
    """Classify a single velocity and acceleration point."""
    # Compute likelihoods
    bird_velocity_likelihood = bird_func.prob(velocity)
    plane_velocity_likelihood = plane_func.prob(velocity)

    bird_acceleration_likelihood = bird_func.prob(acceleration)
    plane_acceleration_likelihood = plane_func.prob(acceleration)

    # Combine likelihoods
    bird_likelihood = bird_velocity_likelihood * bird_acceleration_likelihood
    plane_likelihood = plane_velocity_likelihood * plane_acceleration_likelihood

    # Apply priors
    bird_posterior = bird_likelihood * 0.5
    plane_posterior = plane_likelihood * 0.5

    # Apply transition probabilities
    if prev_class is not None:
        if prev_class == "bird":
            bird_posterior *= transition_probs["bird_to_bird"]
            plane_posterior *= transition_probs["bird_to_plane"]
        elif prev_class == "plane":
            bird_posterior *= transition_probs["plane_to_bird"]
            plane_posterior *= transition_probs["plane_to_plane"]

    # Normalize posteriors
    total_posterior = bird_posterior + plane_posterior
    bird_posterior /= total_posterior
    plane_posterior /= total_posterior

    # Classify based on higher posterior
    return ("bird", bird_posterior) if bird_posterior > plane_posterior else ("plane", plane_posterior)

def classify_track(velocities, accelerations, transition_probs):
    """Classify an entire track of velocity and acceleration data."""
    classifications = []
    prev_class = None

    for velocity, acceleration in zip(velocities, accelerations):
        current_class, _ = classify_point(velocity, acceleration, prev_class, transition_probs)
        classifications.append(current_class)
        prev_class = current_class  # Update for next point

    # Final classification based on majority vote
    final_class = max(set(classifications), key=classifications.count)
    return classifications, final_class

def classify_all_tracks(velo_data, accel_data, transition_probs):
    """Classify all tracks in the dataset."""
    results = []

    for i in range(velo_data.shape[0]):  # Loop through tracks
        sample_classes, final_class = classify_track(velo_data[i], accel_data[i], transition_probs)
        results.append((i + 1, sample_classes, final_class))

    return results

# Transition probabilities
transition_probs = {
    "bird_to_bird": 0.9,
    "bird_to_plane": 0.1,
    "plane_to_plane": 0.9,
    "plane_to_bird": 0.1
}

# Classify all training tracks
train_results = classify_all_tracks(velo_train, accel_train, transition_probs)

# Print results for training data
print("\nTraining Data Classification Results:")
for track_id, sample_classes, final_class in train_results:
    print(f"Track {track_id}: Final Class = {final_class}")
