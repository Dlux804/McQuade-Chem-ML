import os, sys
from padelpy import from_smiles
import pandas as pd
from time import time

"""
Objective: see how long does it take to generate descriptors and fingerprints using PaDEL-descriptor. We'll use let the
code run over all the dataset.
"""
print(sys.path)
# before importing local modules, must add root dir to system path
# capture location of current file (/root/tests/)
myPath = os.path.dirname(os.path.abspath(__file__))
# add to system path the root dir with relative notation: /../ (go up one dir)
sys.path.insert(0, myPath + '/../')

from main import ROOT_DIR
from core import models
from core.misc import cd

# Start timer
start_feat = time()

# Dataset
sets = {

        'water-energy.csv': 'expt'

    }

for data, target in sets.items():  # loop over dataset dictionary
    # change working directory to
    os.chdir(ROOT_DIR)
    with cd('dataFiles'):
        print('Im running ' + str(data))
        model = models.MlModel('rf', data, target)
        # print(model.smiles)
        # Start timer
        start_feat = time()
        features = list(map(from_smiles, model.smiles))
        print(features)
        stop_feat = time()
        feat_time = stop_feat - start_feat
        print('It took '+ feat_time + 'to create features from '+ str(data))


