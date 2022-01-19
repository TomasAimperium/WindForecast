# The Keras model loading function does not play well with
# Pathlib at the moment, so we are using the old os module
# style

import os


PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')

# MODEL PERSISTING
MODEL_NAME = 'lstm'

NEURONS = 80

EPOCHS  = 3

BATCH = 10

DATA_RANGE = [2,100]

DAYS = 10

PCA_N = 4

STATION = "04"