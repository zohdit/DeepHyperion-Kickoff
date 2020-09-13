# Make sure that any of this properties can be overridden using env.properties
import os

# GA Setup
POPSIZE          = int(os.getenv('DH_POPSIZE', '800'))
NGEN             = int(os.getenv('DH_NGEN', '500000'))
ARC_LEN          = POPSIZE

RUNTIME          = int(os.getenv('DH_RUNTIME', '3600'))
INTERVAL         = int(os.getenv('DH_INTERVAL', '900'))

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND    = float(os.getenv('DH_MUTLOWERBOUND', '0.01'))
MUTUPPERBOUND    = float(os.getenv('DH_MUTUPPERBOUND', '0.6'))

# Dataset
EXPECTED_LABEL   = int(os.getenv('DH_EXPECTED_LABEL', '5'))

#------- NOT TUNING ----------

# mutation operator probability
MUTPB            = float(os.getenv('DH_MUTPB', '0.7'))
MUTOPPROB        = float(os.getenv('DH_MUTOPPROB', '0.5'))
MUTOFPROB        = float(os.getenv('DH_MUTOFPROB', '0.5'))


IMG_SIZE         = int(os.getenv('DH_IMG_SIZE', '28'))
num_classes      = int(os.getenv('DH_NUM_CLASSES', '10'))

INITIALPOP       = os.getenv('DH_INITIALPOP', 'seeded')

MODEL            = os.getenv('DH_MODEL', 'models/model_mnist.h5')

ORIGINAL_SEEDS   = os.getenv('DH_ORIGINAL_SEEDS', 'bootstraps_five')

BITMAP_THRESHOLD = float(os.getenv('DH_BITMAP_THRESHOLD', '0.5'))

DISTANCE         = float(os.getenv('DH_DISTANCE', '2.0'))