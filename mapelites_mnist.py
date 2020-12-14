import random
from datetime import datetime

# For Python 3.6 we use the base keras
import keras
import time
import json
# local imports

from mapelites import MapElites
from feature_dimension import FeatureDimension
import utils
import vectorization_tools
from digit_input import Digit

from individual import Individual
from properties import NGEN, \
    POPSIZE, EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, BITMAP_THRESHOLD

from pathlib import Path



# Load the dataset.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Fetch the starting seeds from file
with open(ORIGINAL_SEEDS) as f:
    starting_seeds = f.read().split(',')[:-1]
    random.shuffle(starting_seeds)
    starting_seeds = starting_seeds[:POPSIZE]
    assert (len(starting_seeds) == POPSIZE)


def generate_digit(seed):
    seed_image = x_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    return Digit(xml_desc, EXPECTED_LABEL)


class MapElitesMNIST(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesMNIST, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimension            
        :return: tuple of indexes
        """
        b = tuple()
        for ft in self.feature_dimensions:
            i = ft.feature_descriptor(self, x)
            if i < ft.min:
                ft.min = i
            b = b + (i,)
        return b

    def performance_measure(self, x):
        """
        Apply the fitness function to x
        """
        # "calculate performance measure"    
        pref = x.evaluate()
        return pref

    def mutation(self, x, seed):
        """
        Mutate the solution x
        """
        # "apply mutation"
        Individual.COUNT += 1
        reference = generate_digit(seed)
        x.mutate(reference)
        return x

    def generate_random_solution(self):
        """
        To ease the bootstrap of the algorithm, we can generate
        the first solutions in the feature space, so that we start
        filling the bins
        """
        # "Generate random solution"
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
        digit1.is_original = True
        individual = Individual(digit1, seed)
        individual.seed = seed

        return individual

    def generate_feature_dimensions(self, _type):
        fts = list()
        if _type == 1:
            # feature 1: moves in svg path
            ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance", bins=10)
            fts.append(ft7)

            # feature 2: Number of bitmaps above threshold
            ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
            fts.append(ft2)

        elif _type == 2:
            # feature 1: orientation
            ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc", bins=100)
            fts.append(ft8)

            # feature 2: moves in svg path
            ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance", bins=10)
            fts.append(ft7)

        else:
            # feature 1: orientation
            ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc", bins=100)
            fts.append(ft8)

            # feature 2: Number of bitmaps above threshold
            ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
            fts.append(ft2)

        return fts

    def feature_simulator(self, function, x):
        """
        Calculates the number of control points of x's svg path/number of bitmaps above threshold
        :param x: genotype of candidate solution x
        :return: 
        """
        if function == 'bitmap_count':
            return utils.bitmap_count(x.member, BITMAP_THRESHOLD)
        if function == 'move_distance':
            return utils.move_distance(x.member)
        if function == 'orientation_calc':
            return utils.orientation_calc(x.member, 0)


def main():
    # Generate random folder to store result
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir_name = f"temp_{now}"
    log_dir_name = f"logs/{log_dir_name}"
    # Ensure the folder exists
    Path(log_dir_name).mkdir(parents=True, exist_ok=True)

    print("Logging results to " + log_dir_name)

    for i in range(1, 4):
        map_E = MapElitesMNIST(i, NGEN, POPSIZE, log_dir_name, True)
        map_E.run()


        Individual.COUNT = 0


if __name__ == "__main__":
    main()
