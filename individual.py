import numpy as np
import keras

import evaluator
import predictor
from properties import EXPECTED_LABEL, num_classes
from utils import get_distance
from digit_mutator import DigitMutator

class Individual(object):
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()
    COUNT_MISS = 0

    def __init__(self, member, seed):
        self.seed = seed
        self.ff = None
        self.member = member

    def reset(self):
        self.ff = None

    def evaluate(self):
        if self.ff is None:          
            self.member.predicted_label, self.member.P_class, self.member.P_notclass = \
                predictor.Predictor.predict(self.member.purified)

            # Calculate fitness function
            self.ff = evaluator.evaluate_ff(self.member.P_class, self.member.P_notclass)
            
        return self.ff

    def mutate(self, reference):
        DigitMutator(self.member).mutate(reference)       
        self.reset()

        
