import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


np.random.seed(0)
data_size = 300

# some generic feature
gender = np.random.binomial(1, .5, data_size)
age = stats.truncnorm.rvs((18-30)/25, (80-30)/25, loc=30, scale=25, size=data_size)
car_cost = stats.truncnorm.rvs((10000-39000)/10000, (400000-39000)/10000, loc=39000, scale=10000, size=data_size)
miles = stats.truncnorm.rvs((0-100000)/50000, (300000-100000)/50000, loc=100000, scale=50000, size=data_size)
brand = np.random.uniform(0, 100, data_size)
# some random feature
random_feature = np.zeros((5, data_size))
random_feature[0] = np.random.binomial(10, .7, data_size)
random_feature[1] = np.random.uniform(50, 100, data_size)
random_feature[2] = np.random.normal(0, 1, data_size)
random_feature[3] = np.random.gamma(3, 2, data_size)
random_feature[4] = np.random.standard_t(5, data_size)

# convert features to data
# TODO

