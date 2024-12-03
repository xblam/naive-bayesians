# Ben Lam Implementation of Naive bayesian models

## Description
This is my implementation of naive bayesian models. 

## Guidelines
- essentailly, I am give a pdf of airplanes and birds, relative to the velocity at which they are recorded. 
- I am also given a set of 20 data samples, 10 of planes, and 10 of birds, to train my model on them respectively
- then I am given a dataset of 10 samples, and I must use my bayesian model to guess whether they are recordings of planes or recordings of birds.

## Code Justification and Pitfalls

# velocity pdfs
- my first attempt was to see if I could just classify them by running a naive bayesian on their velocities, by using the provided velocity pdf. This attemp did not work as there was not enough fidelity in the dataset to seperate the models

# Acceleration pdfs
- my second idea was to generate acceleration pdfs from the given dataset. By differentiating through the velocity recordings (after dealing with nans) I could generate a pdf of acceleration values, and use that to refer to as well as velocity
- however, when running naive bayesian with the acceleration and velocity factored into the likelihood, it turns out that the model was still unabel to differentiate between birds and planes.

# Fourier transforms
- from observing how birds 'occilate' their velocities much more frequently than planes, I though I could make a statistic that tracks this to differentiate between birds and planes. I wrote up a fourier analysis class that could:
1. create smooth pdfs
2. allow me to track occilations through the decomposed waves
- however, due to my lack on familiarity with fourier transforms, I was unable to use this function to any significant effect.

# Ending solution
- my ending solution is to simply run the naive bayesian on the acceleration pdf. With this combination, I am able to classify correctly 8 out of 10 objects, which was markedly better than any of my attempts at a more complex way of running the naive bayesian.

# Hypothesized issue
- i believe the greatest weakness of my current setup lies in how the distributions of planes and birds are too overlapped, so if there is a slow plane, the bayesian model will always classify it as a bird if I just use velocity values
- likewise, since the acceleration values for planes is so spread out (centered around 0, but then there are some very large outliers), I am unable to effectly use accelerations to "seperate" the values in the bayesian likelihood calculation. Unfortunately, combining these two statistics also does not tend to help, as they are typically of different magnitudes, and their combination is rendered uneffective. 
- I was thinking of normalizing the individual likelihoods so that prob_v(plane) and prob_v(bird) added to 1 for both velocity and acceleration, but this also did not show any significant improvement.


## Run Notes
- When ran, all of the data will automatically be processed, and the final results of the classification will be printed out in terminal.
- I will almost certainly revisit this project someday to look over where the clasffications might have gone wrong, and to finally fully implement my fourier transforms to track occilations.