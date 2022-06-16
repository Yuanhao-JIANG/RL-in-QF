import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


class Env:
    def __init__(self, glm, df):
        self.glm = glm

        add_group(df)
        self.df = df

    def reset(self):
        return count_group(self.df)

    def step(self, c):
        """
        promote with a price
        :param c: promoted price
        :return: reward, the (grouped) dataframe after promotion
        """

        # add price to dataframe
        self.df['price'] = c

        # glm generated response -> 1: buyer, 0: won't buy
        pred = self.glm.predict(self.df).gt(0.5).astype(int)  # <class 'pandas.core.series.Series'>

        # add glm generated response to dataframe
        self.df['response'] = pred

        # total profit
        r1 = (pred * (c - (self.df['car_cost'] / 300000 + self.df['rand_feature_0'] / 10) * c)).sum()

        # group buyers percentage, there are 4 groups
        k = 4
        p = np.zeros(k)
        for i in range(k):
            p[i] = self.df.groupby(['group', 'response']).size()[i][1] / self.df.groupby(['group']).size()[i]

        # variance of group buyers percentage, we want balanced buyers count for different groups
        r2 = np.var(p, ddof=1)

        # currently keep df unchanged
        return np.max((0.6 - np.sqrt(r2)) * r1, 0), count_group(self.df)


def add_group(df):
    """
    add a group column to the dataframe
    car_cost\age    <40,    >=40
    <70000          0       2
    >=70000         1       3

    :param df: dataframe to be grouped
    :return: None
    """

    df['group'] = (df['age'] // 40) * 2 + df['car_cost'].__gt__(70000).astype(int)


def count_group(df):
    t = df.groupby(['group']).size()
    return (t/t.sum()).values


def add_observations(df_obs, df):
    """
        add an observation to a grouped dataframe
        :param df_obs: dataframe of observations
        :param df: grouped dataframe
        :return: the concatenated dataframe
        """
    add_group(df_obs)
    return pd.concat([df, df_obs], ignore_index=True)
