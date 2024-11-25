# Ben Lam Implementation of Naive bayesian models

## Description
This is my implementation of naive bayesian models. 

## Guidelines
- essentailly, I am give a pdf of airplanes and birds, relative to the velocity at which they are recorded. 
- I am also given a set of 20 data samples, 10 of planes, and 10 of birds, to train my model on them respectively
- then I am given a dataset of 10 samples, and I must use my bayesian model to guess whether they are recordings of planes or recordings of birds.

## Steps

# fitting line to PDF
- the first big issue encountered was that there was no distribution function provided. Instead, what we got were data points which indicated the probability that objects going at x speed was a bird or plane.
- althoguh rounding the input velocities would have worked fine, it would still be better for us to fit a curve to the datapoints for ease of computation, and for future implementations.
- unfortunately, the functions provided do not follow simple distributions like gaussian or poisson. The best fit was a set of polynomial functions, but this fit was still unsatisfactory.
- but with this in mind, we can use fourier transformations

# Fourier Transformations
- we can use fourier transforms to fit a line to the datapoints.

First I will just make a naive bayesian model, then I will train the model on the 10 airplane and 10 bird samples, and then I will run the trained sample on the actual data and see what kind of results I get in return.


since the speed of birds can never surpass around the 150 speed mark, we can also just say that if an object is observed to be flying above 150km/h we just call it a plane


When ran, this program will give the best score that it can find for each generation that the simulation runs for. User can adjust the number of generations, population, mutation rate, boxes, etc at the top of the program