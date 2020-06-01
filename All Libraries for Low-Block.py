# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:56:42 2020

@author: Home
"""

# || 1. Import Libaries
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import HTML

from parsing_functions import * # -- seperate .py file 
from print_out_functions import * # -- seperate .py file 
from tracab_utils import * # -- seperate .py file 
from event_utils import * # -- seperate .py file 
from speed_functions import * # -- seperate .py file 

import pandas as pd
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlrose
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier



# Display options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 3000

# Display options for numpy
np.set_printoptions(linewidth=120, suppress=True)

# Display options for pyplot
%config InlineBackend.figure_format = 'retina'

## Cell computation timer
#%load_ext autotime

## other imports 
from pitch import Pitch # -- seperate .py file 

## KDE specific 
import scipy
import scipy.stats as st
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!