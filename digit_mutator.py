import random
import numpy as np

import mutation_manager
import rasterization_tools
import vectorization_tools
from digit_input import Digit
from properties import MUTOPPROB, EXPECTED_LABEL, MUTOFPROB, DISTANCE
from utils import get_distance


class DigitMutator:

    def __init__(self, digit):
        self.digit = digit

    def mutate(self, reference):       
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            rand_mutation_probability = random.uniform(0, 1)
            rand_mutation_prob = random.uniform(0, 1)
            if rand_mutation_probability >= MUTOPPROB:            
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 1
                else:
                    mutation = 2
            else:
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 3
                else:
                    mutation = 4

            counter_mutations += 1
            mutant_vector = mutation_manager.mutate(self.digit.xml_desc, mutation, counter_mutations/20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_digit = rasterization_tools.rasterize_in_memory(mutant_xml_desc)
            
            distance_inputs = get_distance(reference.purified, rasterized_digit)
            
            if distance_inputs != 0 and distance_inputs <= DISTANCE:
                condition = False    
    
        self.digit.xml_desc = mutant_xml_desc
        self.digit.purified = rasterized_digit   
        self.digit.is_original = False     
        

