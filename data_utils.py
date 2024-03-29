import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

feature_size = 16


def get_truncated_normal(mean, std, low, up, size):
    return stats.truncnorm((low - mean) / std, (up - mean) / std, loc=mean, scale=std).rvs(size)


# generate a customer as numpy array
def generate_customer(seed=None):
    if seed is not None:
        np.random.seed(seed)

    customer = np.zeros(feature_size)

    # customer features:
    # gender
    customer[0] = np.random.binomial(1, .5, 1)

    # age
    customer[1] = get_truncated_normal(mean=25, std=20, low=18, up=90, size=1)

    # car cost
    customer[2] = get_truncated_normal(mean=23000, std=10000, low=10000, up=50000, size=1)

    # miles
    customer[3] = get_truncated_normal(mean=1000, std=4000, low=1, up=250000, size=1)

    # brand
    customer[4] = np.random.uniform(0, 100, 1)

    # some random features:
    customer[5] = np.random.binomial(10, .7, 1)
    customer[6] = np.random.uniform(50, 100, 1)
    customer[7] = np.random.normal(0, 1, 1)
    customer[8] = np.random.gamma(3, 2, 1)
    customer[9] = np.random.standard_t(5, 1)
    customer[10] = np.random.normal(0, 1, 1)
    customer[11] = np.random.normal(1, 2, 1)
    customer[12] = np.random.normal(1.5, 5, 1)
    customer[13] = np.random.normal(1, 2, 1)
    customer[14] = np.random.normal(2, 4, 1)

    # level: 0 -> poor, 1 -> medium, 2 -> rich
    customer[15] = np.random.choice(3, 1)[0]

    return customer


# generate a bunch of customers as dataframe to fit the glm
def generate_dataframe(data_size, save=False, path='./data/dataframe.csv', seed=None):
    if seed is not None:
        np.random.seed(seed)

    feature_data = np.zeros((feature_size, data_size))
    cols = []

    # customer features:
    # gender
    feature_data[0] = np.random.binomial(1, .5, data_size)
    cols.append('gender')

    # age
    feature_data[1] = get_truncated_normal(mean=25, std=20, low=18, up=90, size=data_size)
    cols.append('age')

    # car cost
    feature_data[2] = get_truncated_normal(mean=23000, std=10000, low=10000, up=50000, size=data_size)
    cols.append('car_cost')

    # miles
    feature_data[3] = get_truncated_normal(mean=1000, std=4000, low=1, up=250000, size=data_size)
    cols.append('miles')

    # brand
    feature_data[4] = np.random.uniform(0, 100, data_size)
    cols.append('brand')

    # some random features:
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
    for i in range(10):
        cols.append(f'rand_feature_{i}')

    # level: 0 -> poor, 1 -> medium, 2 -> rich
    feature_data[15] = np.random.choice(3, data_size)
    cols.append('level')

    # price
    price = get_truncated_normal(mean=800, std=600, low=200, up=2000, size=data_size)
    cols.append('price')

    # response
    response = \
        - 0.3 * (feature_data[1] - feature_data[1].mean()) / (feature_data[1].std() + 1e-10) \
        + 0.9 * (feature_data[2] - feature_data[2].mean()) / (feature_data[2].std() + 1e-10) \
        - 0.5 * (feature_data[3] - feature_data[3].mean()) / (feature_data[3].std() + 1e-10) \
        - 0.2 * (feature_data[5] - feature_data[5].mean()) / (feature_data[5].std() + 1e-10) \
        + 0.2 * (feature_data[6] - feature_data[6].mean()) / (feature_data[6].std() + 1e-10) \
        - 0.1 * (feature_data[10] - feature_data[10].mean()) / (feature_data[10].std() + 1e-10) \
        + 0.4 * (feature_data[11] - feature_data[11].mean()) / (feature_data[11].std() + 1e-10) \
        - 0.2 * (feature_data[12] - feature_data[12].mean()) / (feature_data[12].std() + 1e-10) \
        + (feature_data[15] - feature_data[15].mean()) / (feature_data[15].std() + 1e-10) \
        - 1.6 * (price - price.mean()) / (price.std() + 1e-10)
    response = 1 / (1 + np.exp(- response))
    response = np.random.binomial(1, response)
    cols.append('response')

    price = np.array([price])
    response = np.array([response])
    data_np = np.concatenate((feature_data.T, price.T, response.T), axis=1)
    data_df = pd.DataFrame(data_np, columns=cols)
    if save:
        data_df.to_csv(path, index=False)
    return data_df


def fit_glm(fit_df_path='./data/dataframe_fit.csv', save=False, path='./data/glm.model'):
    df_fit = pd.read_csv(fit_df_path)
    formula = 'response ~ gender + age + car_cost + miles + brand + rand_feature_0 + rand_feature_1 ' \
              '+ rand_feature_2 + rand_feature_3 + rand_feature_4 + rand_feature_5 + rand_feature_6 ' \
              '+ rand_feature_7 + rand_feature_8 + rand_feature_9 + level + price'
    glm_raw = smf.glm(formula=formula, data=df_fit, family=sm.families.Binomial())
    glm_fit = glm_raw.fit()
    if save:
        glm_fit.save(path)
    return glm_fit


def plot_csv(models, indices, csv_paths=None, save_path=None):
    dfs = [pd.read_csv('data/' + model + '_out.csv', header=None) for model in models] if not csv_paths \
        else [pd.read_csv(csv_path) for csv_path in csv_paths]
    fig, axs = plt.subplots(len(models), figsize=(10, len(models) * 4))
    fig.supxlabel('Iteration')
    fig.supylabel('Moving Average Reward')
    ys = [dfs[i].iloc[indices[i]] for i in range(len(models))]
    xs = [range(len(y)) for y in ys]
    if len(models) == 1:
        axs.set_title('Moving Average Reward with ' + models[0])
        axs.plot(xs[0], ys[0])
    else:
        for i in range(len(axs)):
            axs[i].set_title('Moving Average Reward with ' + models[i])
            axs[i].plot(xs[i], ys[i])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
