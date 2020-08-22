# GA Setup
POPSIZE = 800
NGEN = 5000
ARC_LEN = POPSIZE

RUNTIME = 3600
INTERVAL = 900

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

K_SD = 0.1

# K-nearest
K = 1

# Dataset
EXPECTED_LABEL = 5

#------- NOT TUNING ----------

# mutation operator probability
MUTPB = 0.7
MUTOPPROB = 0.5
MUTOFPROB = 0.5


IMG_SIZE = 28
num_classes = 10

FITNESS = 'missclassify' #or suprise

INITIALPOP = 'seeded'

MODEL = 'models/model_mnist.h5'

ORIGINAL_SEEDS = "bootstraps_five"

RESULTS_PATH = 'solution'

BITMAP_THRESHOLD = 0.5