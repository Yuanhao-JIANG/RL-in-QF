import pandas as pd
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import statsmodels.api as sm
import env


df_train = pd.read_csv('./data/dataframe_train.csv')
