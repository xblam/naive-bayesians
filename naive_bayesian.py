import numpy as np

# Read and convert dataset.txt
with open("dataset.txt", "r") as file:
    dataset_array = [line.strip().split() for line in file]  # Strip and split each line
    dataset_array = np.array(dataset_array, dtype=float)  # Convert to NumPy array (if numeric)

print(f"dataset shape : {dataset_array.shape}")

with open("testing.txt", "r") as file:
    testing_array = [line.strip().split() for line in file]  # Strip and split each line
    testing_array = np.array(testing_array, dtype=float)  # Convert to NumPy array (if numeric)

print(f"testing shape : {testing_array.shape}")

with open("likelihood.txt", "r") as file:
    likelihood_array = [line.strip().split() for line in file]  # Strip and split each line
    likelihood_array = np.array(likelihood_array, dtype=float)  # Convert to NumPy array (if numeric)

print(f"likelihood shape : {likelihood_array.shape}")



