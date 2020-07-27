"""
Objective: Analyze machine learning results by making graphs
"""


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from time import time
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from rdkit.Chem import PandasTools


