import numpy as np

# easiest way to load up data into numpy
train = np.loadtxt("dataset.txt")

mid = len(train)//2
print(mid)

bird_data = 