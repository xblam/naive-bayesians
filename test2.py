import numpy as np
import matplotlib.pyplot as plt
# first need to get a way to get the function for acceleration from velocity

velo_prob = np.loadtxt('dataset.txt')
# print(velo_prob.shape)

bird_velo_prob = velo_prob[0]
plane_velo_prob = velo_prob[11]


def compute_acceleration(velo_prob, velocities=None):
    # Default to [0, 1, 2, ..., 399] if velocities are not provided
    if velocities is None:
        velocities = np.arange(len(velo_prob))

    # Compute acceleration as the numerical derivative of velocity probabilities
    accelerations = np.gradient(velo_prob, velocities)

    return velocities, accelerations

bird_velo, bird_accel = compute_acceleration(bird_velo_prob)
plane_velo, plane_accel = compute_acceleration(plane_velo_prob)



# Bird original velocity, probability, and acceleration
plt.subplot(2, 2, 1)
plt.plot(bird_original_velo, bird_velo_prob, label="Bird Velocity Probability", color="blue")
plt.plot(bird_original_velo, bird_accel, label="Bird Acceleration", color="cyan")
plt.xlabel("Velocity")
plt.ylabel("Probability / Acceleration")
plt.title("Bird: Velocity Probabilities and Acceleration")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(bird_original_velo, bird_original_velo, label="Original Bird Velocity", color="green")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Bird: Original Velocity Data")
plt.legend()
plt.grid()

# Plane original velocity, probability, and acceleration
plt.subplot(2, 2, 3)
plt.plot(plane_original_velo, plane_velo_prob, label="Plane Velocity Probability", color="orange")
plt.plot(plane_original_velo, plane_accel, label="Plane Acceleration", color="red")
plt.xlabel("Velocity")
plt.ylabel("Probability / Acceleration")
plt.title("Plane: Velocity Probabilities and Acceleration")
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(plane_original_velo, plane_original_velo, label="Original Plane Velocity", color="purple")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Plane: Original Velocity Data")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()