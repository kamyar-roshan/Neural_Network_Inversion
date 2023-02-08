### import libraries
import os
import sys
import math
import copy
import time
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from scipy.misc import derivative
from sklearn import preprocessing
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)   

### path to the python script for data generation
Path2script = "/storage/work/kar/ForceField_Optimization/NNI/"
sys.path.append(Path2script)
warnings.filterwarnings("ignore")

### import modules from ffield generator folder
import ffield_generator
from ffield_generator.error_output import error_output
from ffield_generator.remove_forts import remove_forts
from ffield_generator.parameter import assign_parameter
from ffield_generator.define_ranges import define_ranges
from ffield_generator.params_input_train import params_input_train
from ffield_generator.generate_training_folder import generate_training_folder