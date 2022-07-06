import pandas as pd
import numpy as np
import torch
import generate_data
import statsmodels.api as sm
import statsmodels.formula.api as smf


class Env:
    def __init__(self, glm):
        self.glm = glm
        self.df = None
        self.current_customer = None
        self.state = None
        self.current_customer_df = None

        cols = ['gender', 'age', 'car_cost', 'miles', 'brand']
        for i in range(10):
            cols.append(f'rand_feature_{i}')
        cols.append('price')
        cols.append('response')
        cols.append('profit')
        cols.append('group')
        self.cols = cols

    def get_new(self):
        self.current_customer = generate_data.generate_customer()
        # groups:
        # car_cost\age    <40,    >=40
        # <70000          0       2
        # >=70000         1       3
        group = (self.current_customer['age'] // 40) * 2 + self.current_customer['car_cost'].__gt__(70000).astype(int)
        self.current_customer_df = pd.DataFrame(data=np.append(self.current_customer, [0, 0, 0, group]),
                                                columns=self.cols)

    def reset(self):
        self.get_new()
        self.state = np.append(self.current_customer, [0, 0, 0])
        self.df = pd.DataFrame(columns=self.cols)

        return self.state

    def step(self, c):
        # promote price to customer
        self.current_customer_df['price'] = c

        # glm generated response -> 1: buyer, 0: won't buy, and add this response to this customer's record
        pred = self.glm.predict(self.current_customer_df).gt(0.5).astype(int)
        self.current_customer_df['response'] = pred

        # compute customer profit
        pf = pred * (c - (self.current_customer['car_cost']/300000 + self.current_customer['rand_feature_0']/10) * c)
        self.current_customer_df['profit'] = pf

        # add the customer to the dataframe,
        # then compute the average profit, the buyers portion, and the variance of the groups
        pd.concat([self.df, self.current_customer_df], ignore_index=True)
        avg_pf = self.df['profit'].mean()
        p = self.df['response'].mean()
        var = count_var(self.df)

        # generate new customer and state
        self.get_new()
        self.state = np.append(self.current_customer, [avg_pf, p, var])

        # return reward + next state
        return np.max(pf*(1 - np.sqrt(var)/2), 0), self.state


# group buyers percentage, there are 4 groups
def count_var(df):
    g = list(map(int, df.groupby(['group']).groups.keys()))
    p = np.zeros(4)
    for i in range(len(g)):
        choice = list(map(int, df.groupby(['group', 'response']).size()[i].keys()))
        if choice[0] == 1:
            buyers = df.groupby(['group', 'response']).size()[i][0]
        elif len(choice) == 2:
            buyers = df.groupby(['group', 'response']).size()[i][0]
        else:
            buyers = 0

        p[g[i]] = buyers / df.groupby(['group']).size()[i]

    var = np.var(p)
    return var
