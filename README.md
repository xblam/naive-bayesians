# Ben Lam Implementation of Naive bayesian models

## Description
This is my implementation of naive bayesian models. 

## Guidelines
- essentailly, I am give a pdf of airplanes and birds, relative to the velocity at which they are recorded. 
- I am also given a set of 20 data samples, 10 of planes, and 10 of birds, to train my model on them respectively
- then I am given a dataset of 10 samples, and I must use my bayesian model to guess whether they are recordings of planes or recordings of birds.

## Steps

# Preprocessing
- formatted the data into np arrays (for ease of computation) and split the nessecary datasets into bird and plane parts
- initial reaction was just to fit a gaussian distribution to the data, but this was not easy to do
- since I have been working with fourier transformations recently I thought it would be a good idea to get a re

First I will just make a naive bayesian model, then I will train the model on the 10 airplane and 10 bird samples, and then I will run the trained sample on the actual data and see what kind of results I get in return.


since the speed of birds can never surpass around the 150 speed mark, we can also just say that if an object is observed to be flying above 150km/h we just call it a plane


When ran, this program will give the best score that it can find for each generation that the simulation runs for. User can adjust the number of generations, population, mutation rate, boxes, etc at the top of the program