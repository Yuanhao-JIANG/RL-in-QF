import pandas as pd
import numpy as np
import data_utils


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
        cols.append('level')

        cols.append('price')
        cols.append('response')
        cols.append('profit')
        cols.append('group')
        self.cols = cols

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
        self.get_new()
        self.state = np.append(self.current_customer, [0, 0, 0])
        self.df = pd.DataFrame(columns=self.cols)

        return self.state

    def g(self):
        return self.current_customer[2]/60 + 10*self.current_customer[5]

    def h(self, var):
        return np.sqrt(var)/2

    def step(self, c):
        # promote price to customer
        self.current_customer_df['price'] = c

        # glm generated response -> 1: buyer, 0: won't buy, and add this response to this customer's record
        pred = self.glm.predict(self.current_customer_df).gt(0.5).astype(int)[0]
        self.current_customer_df['response'] = pred

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
        self.state = np.append(self.current_customer, [avg_pf, p, var])

        # return reward + next state
        return pf*(1 - self.h(var)), self.state

    # compute expected reward and expected next state, but not step to it
    def peek_expected(self, c):
        # promote price to customer
        self.current_customer_df['price'] = c

        # glm generated response -> 1: buyer, 0: won't buy, and add this response to this customer's record
        pred = self.glm.predict(self.current_customer_df).gt(0.5).astype(int)[0]
        self.current_customer_df['response'] = pred

        # compute customer profit
        pf = pred * (c - self.g())
        self.current_customer_df['profit'] = pf

        # add the customer to the dataframe,
        # then compute the average profit, the buyers portion, and the variance of the groups
        self.df = pd.concat([self.df, self.current_customer_df], ignore_index=True)
        avg_pf = self.df['profit'].mean()
        p = self.df['response'].mean()
        var = count_var(self.df)

        self.df = self.df.iloc[:-1]

        # generate expected customer and state
        expected_customer = np.array([.5, 28, 39000, 8000, 50, 7, 75, 0, 6, 0, 0, 1, 1.5, 1, 2])
        expected_state = np.append(expected_customer, [avg_pf, p, var])

        # return reward + next state
        return pf * (1 - self.h(var)), expected_state


# group buyers percentage, there are 4 groups
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
