import operator
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import json

# local imports
from individual import Individual
from plot_utils import plot_heatmap, plot_fives
import utils
from properties import RUNTIME, INTERVAL


class MapElites(ABC):

    def __init__(self, _type, iterations, bootstrap_individuals, log_dir_path, minimization):
        """
        :param iterations: Number of evolutionary iterations        
        :param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm       
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.elapsed_time = 0
        self.log_dir_path = log_dir_path
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.iterations = iterations

        self.random_solutions = bootstrap_individuals
        self.feature_dimensions = self.generate_feature_dimensions(_type)

        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = np.full(
            ft_bins, None,
            dtype=object
        )
        self.performances = np.full(ft_bins, np.inf, dtype=float)

        print("Configuration completed.")

    def generate_initial_population(self):
        """
        Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals
        randomly sampled from a uniform distribution
        """
        print("Generate initial population")
        for _ in range(0, self.random_solutions):
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x)

    def run(self):
        """
        Main iteration loop of MAP-Elites
        """
        start_time = datetime.now()
        # start by creating an initial set of random solutions
        self.generate_initial_population()
        ii = 1
        for i in range(0, self.iterations):
            elapsed_time = datetime.now() - start_time
            if elapsed_time.seconds <= RUNTIME:
                print(f"ITERATION {i}")
                print("Select and mutate.")
                # get the index of a random individual from the map of elites
                ind = self.random_selection(individuals=1)[0]
                # mutate the individual
                ind = self.mutation(ind, ind.seed)
                # place the new individual in the map of elites
                self.place_in_mapelites(ind)
                elapsed_time = datetime.now() - start_time
                if elapsed_time.seconds >= self.config.INTERVAL * ii:
                    self.extract_results(i, (self.config.INTERVAL * ii / 60))
                    ii += 1

        self.extract_results(i, (INTERVAL * ii / 60))

        end_time = datetime.now()
        self.elapsed_time = end_time - start_time

        if self.minimization:
            best = self.performances.argmin()
        else:
            best = self.performances.argmax()

        idx = np.unravel_index(best, self.performances.shape)
        best_perf = self.performances[idx]
        best_ind = self.solutions[idx]
        print(f"Best overall value: {best_perf}"
              f" produced by individual {best_ind}"
              f" and placed at {self.map_x_to_b(best_ind)}")

    def extract_results(self, iterations, execution_time):
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        # self.log_dir_path is "logs/temp_..."
        log_dir_name = f"{self.log_dir_path}/log_{self.random_solutions}_{iterations}_{execution_time}_{now}"

        # Create another folder insider the log one ...
        log_dir_path = Path(f"{log_dir_name}/{self.feature_dimensions[1].name}_{self.feature_dimensions[0].name}")
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # filled values                                 
        filled = np.count_nonzero(self.solutions != None)
        total = np.size(self.solutions)        

        original_seeds = set()
        mis_seeds = set()
        for (i, j), value in np.ndenumerate(self.solutions):
            if self.solutions[i, j] is not None:
                original_seeds.add(self.solutions[i, j].seed)
                if self.performances[i, j] < 0:
                    mis_seeds.add(self.solutions[i, j].seed)

        Individual.COUNT_MISS = 0
        for (i, j), value in np.ndenumerate(self.performances):
            if self.performances[i, j] < 0:
                Individual.COUNT_MISS += 1
                utils.print_image(f"{log_dir_path}/({i},{j})", self.solutions[i, j].member.purified, '')
            elif 0 < self.performances[i, j] < np.inf:
                utils.print_image(f"{log_dir_path}/({i},{j})", self.solutions[i, j].member.purified, 'gray')

        report = {
            'Covered seeds': len(original_seeds),
            'Filled cells': (filled),
            'Filled density': (filled / total),
            'Misclassified seeds': len(mis_seeds),
            'Misclassification': (Individual.COUNT_MISS),
            'Misclassification density': (Individual.COUNT_MISS / filled),
            'Performances': self.performances.tolist()
        }
        
        dst = f"{self.log_dir_path}/report_" + self.feature_dimensions[1].name + "_" + self.feature_dimensions[
            0].name + "_" + str(execution_time) + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()
        
        self.plot_map_of_elites(self.performances, log_dir_name)    
        plot_fives(f"{log_dir_name}", self.feature_dimensions[1].name, self.feature_dimensions[0].name)    
        

    def place_in_mapelites(self, x):
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:

        - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
        - Compute the performance of the solution
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        :param x: genotype of an individual
        """
        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        # performance of the x
        perf = self.performance_measure(x)

        reconstruct = False
        for i in range(len(b)):
            if b[i] >= self.feature_dimensions[i].bins:
                reconstruct = True
                self.feature_dimensions[i].bins = b[i] + 1

        if reconstruct:
            self.reconstruct_map()

        # place operator performs either minimization or maximization
        if self.place_operator(perf, self.performances[b]):
            print(f"PLACE: Placing individual {x} at {b} with perf: {perf}")
            self.performances[b] = perf
            self.solutions[b] = x
        else:
            print(f"PLACE: Individual {x} rejected at {b} with perf: {perf} in favor of {self.performances[b]}")

    def random_selection(self, individuals=1):
        """
        Select an elite x from the current map of elites.
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        :param individuals: The number of individuals to randomly select
        :return: A list of N random elites
        """

        def _get_random_index():
            """
            Get a random cell in the N-dimensional feature space
            :return: N-dimensional tuple of integers
            """
            indexes = tuple()
            for ft in self.feature_dimensions:
                rnd_ind = np.random.randint(0, ft.bins, 1)[0]
                indexes = indexes + (rnd_ind,)
            return indexes

        def _is_not_initialized(index):
            """
            Checks if the selected index points to a None solution (not yet initialized)            
            :return: Boolean
            """
            if self.solutions[index] == None:
                return True
            return False

        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_random_index()
            # we do not want to repeat entries
            while idx in idxs or _is_not_initialized(idx):
                idx = _get_random_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return inds

    def get_elapsed_time(self):
        return self.elapsed_time

    def reconstruct_map(self):
        """
        Extend Map of elites dynamically if needed
        """
        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        new_solutions = np.full(
            ft_bins, None,
            dtype=(object)
        )
        new_performances = np.full(ft_bins, np.inf, dtype=float)

        new_solutions[0:self.solutions.shape[0], 0:self.solutions.shape[1]] = self.solutions
        new_performances[0:self.performances.shape[0], 0:self.performances.shape[1]] = self.performances
        self.solutions = new_solutions
        self.performances = new_performances
        return

    def plot_map_of_elites(self, perfs, log_dir_name):
        """
        Plot a heatmap of elites
        """
        plot_heatmap(perfs,
                     self.feature_dimensions[1].name,
                     self.feature_dimensions[0].name,
                     savefig_path=log_dir_name
                     )

    @abstractmethod
    def performance_measure(self, x):
        """
        Function to evaluate solution x and give a performance measure
        :param x: genotype of a solution
        :return: performance measure of that solution
        """
        pass

    @abstractmethod
    def mutation(self, x, reference):
        """
        Function to mutate solution x and give a mutated solution
        :param x: genotype of a solution
        :return: mutated solution
        """
        pass

    @abstractmethod
    def map_x_to_b(self, x):
        """
        Function to map a solution x to feature space dimensions
        :param x: genotype of a solution
        :return: phenotype of the solution (tuple of indices of the N-dimensional space)
        """
        pass

    @abstractmethod
    def generate_random_solution(self):
        """
        Function to generate an initial random solution x
        :return: x, a random solution
        """
        pass

    @abstractmethod
    def generate_feature_dimensions(self, _type):
        """
        Generate a list of FeatureDimension objects to define the feature dimension functions
        :return: List of FeatureDimension objects
        """
        pass
