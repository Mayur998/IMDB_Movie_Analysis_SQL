# IMDB_Movie_Analysis_SQL
In this case study we are doing analysis on IMDB movie analysis and get the insights from data.

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing.


# Import necessary libraries
from deap import base, creator, tools
import random
import numpy as np

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
  # Use the individual as the input to the model and return the output
  output = model(individual)
  return output,

# Define the bounds for the input values
bounds = [(-5, 5), (-5, 5), (-5, 5)]  # for example

# Create a DEAP creator object for the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)  # for example
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set the parameters for the genetic algorithm
pop_size = 50
n_generations = 100

# Initialize the population
pop = toolbox.population(n=pop_size)

# Evaluate the fitness of the population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Begin the evolution
for g in range(n_generations):
    # Select the next generation of individuals
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate the fitness of the offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Replace the current population with the offspring
    pop[:] = offspring

# Find the individual with the highest fitness value
best_ind = tools.selBest(pop, k=1)[0]

# Print the input values of the best individual
print('Best input values:', best
