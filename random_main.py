import itertools
import random
import time
from datetime import datetime
import sys

import numpy as np
from deap import base, creator, tools
from deap.tools import selTournament
import keras
from pathlib import Path
import json

import archive_manager
import vectorization_tools
from digit_input import Digit
from digit_mutator import DigitMutator
import plot_utils
import utils
from individual import Individual
from properties import NGEN, \
    EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, RUNTIME, INTERVAL
from mapelites_mnist import MapElitesMNIST

POPSIZE = int(sys.argv[1])

# Load the dataset.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Fetch the starting seeds from file
with open(ORIGINAL_SEEDS) as f:
    starting_seeds = f.read().split(',')[:-1]
    random.shuffle(starting_seeds)
    starting_seeds = starting_seeds[:POPSIZE]
    assert(len(starting_seeds) == POPSIZE)

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
# Define the individual.
creator.create("Individual", Individual, fitness=creator.FitnessSingle)


def generate_digit(seed):    
    seed_image = x_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    return Digit(xml_desc, EXPECTED_LABEL)

def generate_individual():
    Individual.COUNT += 1
    if INITIALPOP == 'random':
        # Choose randomly a file in the original dataset.
        seed = random.choice(starting_seeds)
        Individual.SEEDS.add(seed)
    elif INITIALPOP == 'seeded':
        # Choose sequentially the inputs from the seed list.
        # NOTE: number of seeds should be no less than the initial population
        assert (len(starting_seeds) == POPSIZE)
        seed = starting_seeds[Individual.COUNT - 1]
        Individual.SEEDS.add(seed)

    digit1 = generate_digit(seed)
    
    individual = Individual(digit1, seed)    
    return individual


# Evaluate an individual.
def evaluate_individual(individual):
    return individual.evaluate()
    
def mutate_individual(individual):
    Individual.COUNT += 1    
    ref = generate_digit(individual.seed)
    DigitMutator(individual.member).mutate(ref)
    individual.reset()
  
def select_individual(individuals, k):
    chosen = []
    for i in range(k):
        index = random.randint(0,k-1)
        chosen.append(individuals[index])
    return chosen

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", select_individual)
toolbox.register("mutate", mutate_individual)


def main(rand_seed=None):
    start_time = datetime.now()    

    stats = tools.Statistics(lambda ind: ind.ff)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    # Generate initial population.
    population = toolbox.population(n=POPSIZE) 
    invalid_ind = [ind for ind in population]
    
    ii = 1
    # Begin the generational process
    for gen in range(1, NGEN):
        elapsed_time = datetime.now() - start_time
        if elapsed_time.seconds <= RUNTIME:        
            # Vary the population.
            offspring = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Mutation.        
            for mutant in offspring:                      
                toolbox.mutate(mutant)

            invalid_ind = [ind for ind in offspring]            
            fitnesses = [toolbox.evaluate(i) for i in invalid_ind]

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.ff = fit

            for ind in offspring:                                  
                archive.update_archive(ind)

            # Generate maps
            elapsed_time = datetime.now() - start_time
            if (elapsed_time.seconds) >= INTERVAL*ii:                                               
                generate_maps((INTERVAL*ii/60), gen)
                ii += 1 
                        
            # Update the statistics with the new population
            record = stats.compile(offspring)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

            population = offspring
            
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time}")   


def generate_maps(execution_time, iterations):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir_name = f"log_{POPSIZE}_{iterations}_{execution_time}_{now}" 
    log_dir_path = Path(f'random_logs/{log_dir_name}')
    log_dir_path.mkdir(parents=True, exist_ok=True)   
    if len(archive.get_archive()) > 0:
        ''' type #1 : Moves & Bitmaps
            type #2 : Moves & Orientation
            type #3 : Orientation & Bitmaps
        '''
        for i in range(1,4):
            map_E = MapElitesMNIST(i, NGEN, POPSIZE, True)                           
            log_dir_path = Path(f'random_logs/{log_dir_name}/{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}')
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            for ind in archive.get_archive():             
                map_E.place_in_mapelites(ind)

            # rescale        
            map_E.solutions, map_E.performances = utils.rescale(map_E.solutions,map_E.performances)     

            # filled values                                 
            filled = np.count_nonzero(map_E.solutions!=None)
            total = np.size(map_E.solutions)
            filled_density = (filled / total)
            
            Individual.COUNT_MISS = 0
            covered_seeds = set()
            for (i,j), value in np.ndenumerate(map_E.performances): 
                if map_E.performances[i,j] != 2.0:
                    covered_seeds.add(map_E.solutions[i,j].seed)
                    if map_E.performances[i,j] < 0: 
                        Individual.COUNT_MISS += 1
                        utils.print_image(f"{log_dir_path}/({i},{j})", map_E.solutions[i,j].member.purified)
                    else:
                        utils.print_image(f"{log_dir_path}/({i},{j})", map_E.solutions[i,j].member.purified, 'gray')

            report = {
                'features': map_E.feature_dimensions[1].name + ',' + map_E.feature_dimensions[0].name,
                'covered_seeds' : len(covered_seeds),
                'filled cells': str(filled),
                'filled density': str(filled_density),
                'misclassification': str(Individual.COUNT_MISS),
                'misclassification density': str(Individual.COUNT_MISS/filled),
                'iterations': str(iterations)
            }  
            map_E.log_dir_path = f"random_logs/{log_dir_name}"
            dst = f"random_logs/{log_dir_name}" + f'/report_'+ map_E.feature_dimensions[1].name +'_'+ map_E.feature_dimensions[0].name+'.json'
            report_string = json.dumps(report)

            file = open(dst, 'w')
            file.write(report_string)
            file.close()

            map_E.plot_map_of_elites(map_E.performances)
            plot_utils.plot_fives(map_E.log_dir_path, map_E.feature_dimensions[1].name, map_E.feature_dimensions[0].name)  
            

if __name__ == "__main__":    
    archive = archive_manager.Archive()
    pop = main()
    
    

   

