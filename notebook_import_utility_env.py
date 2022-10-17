"""
%load_ext autoreload
%autoreload 2
import os
prefix = ''
if os.path.expanduser('~') == '/home/baraldian': # UNI env
    prefix = '/home/baraldian'
else:
    from google.colab import drive
    drive.mount('/content/drive')
softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'Concept level EM (exclusive-inclluse words)')

"""

"""
%load_ext autoreload
%autoreload 2
import os
prefix = ''
if os.path.expanduser('~') == '/home/baraldian': # UNI env
    prefix = '/home/baraldian'
else:
    from google.colab import drive
    drive.mount('/content/drive')
softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'Concept level EM (exclusive-inclluse words)')

"""

import os
import socket

host_name = socket.gethostname()
# print(f'{host_name = }')
print(f'{host_name}')
prefix = ''
if os.path.expanduser('~') == '/home/baraldian':  # UNI env
    prefix = '/home/baraldian'
else:
    # install here for colab env
    """!pip install lime
    !pip install -q spacy
    !pip install -q pytorch-lightning
    !pip install -q transformers
    !pip install -q -U sentence-transformers
    !pip install -U nltk
    !pip install pyyaml==5.4.1
    """
    pass

softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'Fairness', 'scalable-fairlearn')

import os, sys, requests, re, ast, pickle, copy, gc, re, time

sys.path.append(os.path.join(project_path, 'common_functions'))
sys.path.append(os.path.join(project_path, 'src'))
sys.path.append(os.path.join(project_path, 'src', 'BERT'))

import pandas as pd
import numpy as np
from warnings import simplefilter
from IPython.utils import io
from tqdm.notebook import tqdm

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore')
pd.options.display.float_format = '{:.4f}'.format
pd.options.display.max_colwidth = 130
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # Display all statements

from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


