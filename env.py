import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def add_group(df):
    """
    add a group column to the dataframe
    car_cost\age    <40,    >=40
    <70000          0       2
    >=70000         1       3

    :param df: dataframe to be grouped
    :return: None
    """

    df['group'] = (df['age'] // 40)*2 + df['car_cost'].__gt__(70000).astype(int)


def add_observations(df_obs, df):
    """
    add an observation to a grouped dataframe
    :param df_obs: dataframe of observations
    :param df: grouped dataframe
    :return: the concatenated dataframe
    """
    add_group(df_obs)
    return pd.concat([df, df_obs], ignore_index=True)


def promote(df_pre, c, glm):
    pass
