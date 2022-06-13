import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


np.random.seed(0)
data_size = 1000
feature_size = 15


def generate_data():
    feature_data = np.zeros((feature_size, data_size))

    # some generic feature:
    # gender
    feature_data[0] = np.random.binomial(1, .5, data_size)
    # age
    age_mean, age_var = 28, 25
    feature_data[1] = stats.truncnorm.rvs((18-age_mean)/age_var, (80-age_mean)/age_var,
                                          loc=age_mean, scale=age_var, size=data_size)
    # car cost
    car_cost_mean, car_cost_var = 39000, 50000
    feature_data[2] = stats.truncnorm.rvs((10000-car_cost_mean)/car_cost_var, (400000-car_cost_mean)/car_cost_var,
                                          loc=car_cost_mean, scale=car_cost_var, size=data_size)
    # miles
    miles_mean, miles_var = 8000, 50000
    feature_data[3] = stats.truncnorm.rvs((200-miles_mean)/miles_var, (250000-miles_mean)/miles_var,
                                          loc=miles_mean, scale=miles_var, size=data_size)
    # brand
    feature_data[4] = np.random.uniform(0, 100, data_size)
    # some random feature:
    feature_data[5] = np.random.binomial(10, .7, data_size)
    feature_data[6] = np.random.uniform(50, 100, data_size)
    feature_data[7] = np.random.normal(0, 1, data_size)
    feature_data[8] = np.random.gamma(3, 2, data_size)
    feature_data[9] = np.random.standard_t(5, data_size)
    feature_data[10] = np.random.normal(0, 1, data_size)
    feature_data[11] = np.random.normal(1, 2, data_size)
    feature_data[12] = np.random.normal(1.5, 5, data_size)
    feature_data[13] = np.random.normal(1, 2, data_size)
    feature_data[14] = np.random.normal(2, 4, data_size)

    feature_data = np.transpose(feature_data)

    # price
    price_mean, price_var = 1400, 700
    price = stats.truncnorm.rvs((400-price_mean)/price_var, (2700-price_mean)/price_var,
                                loc=price_mean, scale=price_var, size=data_size)

    # response
    response = np.zeros(data_size)
    for i in range(data_size):
        t = age_mean/feature_data[i][1] + 1.2*miles_mean/feature_data[i][3] - 4*car_cost_mean/feature_data[i][2] \
            - 2*10*.7/(feature_data[i][5]+1) - (feature_data[i][6] - 49)/50 + 1.5*feature_data[i][10] \
            + 0.5*feature_data[i][11] - 2*feature_data[i][12] - 1.5*feature_data[i][13] + 2.5*price_mean/price[i]
        # 50% normal(0, 1) and 60% t
        if abs(feature_data[i][7]) < 0.67 and abs(feature_data[i][9]) < 0.92:
            if t > 0:
                t *= 1.3
            else:
                t *= 0.4
        # first 70% gamma and outside 10% normal(2, 4)
        if feature_data[i][8] < 7.2 and abs(feature_data[i][14]) > 2.5:
            if t > 0:
                t *= 1.3
            else:
                t *= 0.4
        # response[i] = t
        # response[i] = np.tanh(t/5.5+0.5)
        response[i] = np.random.binomial(1, (np.tanh(t/5.5+0.5) + 1)/2)
        data = [[feature_data[i], price[i], response[i]] for i in range(data_size)]
        data = np.array(data, dtype=object)
        return data


raw_data = generate_data()
np.save('./data/raw_data.npy', raw_data)
