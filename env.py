import pandas as pd
import numpy as np
import data_utils


class Env:
    def __init__(self, glm):
        self.glm = glm
        # dataframe that records all customers information so far, unless reset
        self.df = None
        # current customer being promoted
        self.current_customer = None
        # current state fed or to be fed to the network
        self.state = None
        # current customer being promoted, of type pandas dataframe
        self.current_customer_df = None

        cols = ['gender', 'age', 'car_cost', 'miles', 'brand']
        for i in range(10):
            cols.append(f'rand_feature_{i}')
        cols.append('level')
        cols.append('price')
        cols.append('response')
        cols.append('profit')
        cols.append('group')
        self.cols = cols

    # generate new customer
    def get_new(self):
        self.current_customer = data_utils.generate_customer()
        # groups:
        # car_cost\age    <40,    >=40
        # <70000          0       2
        # >=70000         1       3
        group = (self.current_customer[1] // 40) * 2 + self.current_customer[2].__gt__(70000).astype(int)
        customer = np.append(self.current_customer, [0, 0, 0, group])
        customer = np.expand_dims(customer, axis=0)
        self.current_customer_df = pd.DataFrame(data=customer, columns=self.cols)

    def reset(self):
        self.df = pd.DataFrame(columns=self.cols)
        self.get_new()
        # append average_profit, portion, variance
        self.state = np.concatenate((self.current_customer[:-1], [self.current_customer[-1] == 0,
                                                                  self.current_customer[-1] == 0,
                                                                  self.current_customer[-1] == 0], [0, 0, 0]))
        return self.state

    def g(self):
        return 0

    def h(self, var):
        return np.sqrt(var)/2

    def step(self, c):
        # promote price to customer
        self.current_customer_df['price'] = float(c)

        # glm generated response -> 1: buyer, 0: won't buy, and add this response to this customer's record
        pred = self.glm.predict(self.current_customer_df).gt(0.5).astype(int)[0]
        self.current_customer_df['response'] = float(pred)

        # compute customer profit
        pf = pred * (c - self.g())
        self.current_customer_df['profit'] = pf

        # add the customer to the dataframe,
        # then compute the average profit, the buyers portion, and the variance of the groups
        self.df = pd.concat([self.df, self.current_customer_df], ignore_index=True)
        avg_pf = self.df['profit'].mean()
        p = self.df['response'].mean()
        var = count_var(self.df)

        # generate new customer and state
        self.get_new()
        self.state = np.concatenate((self.current_customer[:-1], [self.current_customer[-1] == 0,
                                                                  self.current_customer[-1] == 0,
                                                                  self.current_customer[-1] == 0], [avg_pf, p, var]))

        # return reward + next state
        return pf*(1 - self.h(var)), self.state


# variance of group buyers percentage, there are 4 groups
def count_var(df):
    p = np.zeros(4)
    group_p = df.groupby('group').mean()['response']
    for i in range(4):
        try:
            p[i] = group_p.at[i]
        except KeyError:
            pass
    var = np.var(p)
    return var
