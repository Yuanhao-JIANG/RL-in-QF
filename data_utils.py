import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from torch.distributions import MultivariateNormal

feature_size = 16


def generate_customer():
    customer = np.zeros(feature_size)

    # some generic feature:
    # gender
    customer[0] = np.random.binomial(1, .5, 1)

    # age
    age_mean, age_var = 28, 25
    customer[1] = stats.truncnorm.rvs((18 - age_mean) / age_var, (80 - age_mean) / age_var,
                                      loc=age_mean, scale=age_var, size=1)
    # car cost
    car_cost_mean, car_cost_var = 39000, 50000
    customer[2] = stats.truncnorm.rvs((10000 - car_cost_mean) / car_cost_var, (400000 - car_cost_mean) / car_cost_var,
                                      loc=car_cost_mean, scale=car_cost_var, size=1)
    # miles
    miles_mean, miles_var = 8000, 50000
    customer[3] = stats.truncnorm.rvs((200 - miles_mean) / miles_var, (250000 - miles_mean) / miles_var,
                                      loc=miles_mean, scale=miles_var, size=1)
    # brand
    customer[4] = np.random.uniform(0, 100, 1)
    # some random feature:
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


def generate_dataframe(data_size=1000, save=False, path='./data/dataframe.csv', seed=0):
    np.random.seed(seed)

    feature_data = np.zeros((feature_size, data_size))
    cols = []

    # customer features:
    # gender
    feature_data[0] = np.random.binomial(1, .5, data_size)
    cols.append('gender')

    # age
    age_mean, age_var = 28, 25
    feature_data[1] = stats.truncnorm.rvs((18 - age_mean) / age_var, (80 - age_mean) / age_var,
                                          loc=age_mean, scale=age_var, size=data_size)
    cols.append('age')

    # car cost
    car_cost_mean, car_cost_var = 39000, 50000
    feature_data[2] = stats.truncnorm.rvs((10000 - car_cost_mean) / car_cost_var,
                                          (400000 - car_cost_mean) / car_cost_var,
                                          loc=car_cost_mean, scale=car_cost_var, size=data_size)
    cols.append('car_cost')

    # miles
    miles_mean, miles_var = 8000, 50000
    feature_data[3] = stats.truncnorm.rvs((200 - miles_mean) / miles_var, (250000 - miles_mean) / miles_var,
                                          loc=miles_mean, scale=miles_var, size=data_size)
    cols.append('miles')

    # brand
    feature_data[4] = np.random.uniform(0, 100, data_size)
    cols.append('brand')

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
    for i in range(10):
        cols.append(f'rand_feature_{i}')

    # level
    feature_data[15] = np.random.choice(3, data_size)
    cols.append('level')

    feature_data = np.transpose(feature_data)

    # price
    price_mean, price_var = 1400, 700
    price = stats.truncnorm.rvs((400 - price_mean) / price_var, (2700 - price_mean) / price_var,
                                loc=price_mean, scale=price_var, size=data_size)
    price = np.array([price])
    cols.append('price')

    # response
    response = np.zeros(data_size)
    for i in range(data_size):
        t = age_mean / feature_data[i][1] + 1.2 * miles_mean / feature_data[i][3] - 4 * car_cost_mean / \
            feature_data[i][2] \
            - 2 * 10 * .7 / (feature_data[i][5] + 1) - (feature_data[i][6] - 49) / 50 + 1.5 * feature_data[i][10] \
            + 0.5 * feature_data[i][11] - 2 * feature_data[i][12] - 1.5 * feature_data[i][13] \
            + (feature_data[i][15] + 1) * price_mean / price[i]
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
        response[i] = np.random.binomial(1, (np.tanh(t / 5.5 + 0.5) + 1) / 2)
    response = np.array([response])
    cols.append('response')

    data = np.concatenate((feature_data, price.T, response.T), axis=1)
    df = pd.DataFrame(data, columns=cols)
    if save:
        df.to_csv(path, index=False)
    return df


def fit_glm(fit_df_path='./data/dataframe_fit.csv', save=False, path='./data/glm.model'):
    df_fit = pd.read_csv(fit_df_path)
    formula = 'response ~ gender'
    for i in range(len(df_fit.columns) - 2):
        formula += f' + {df_fit.columns[i + 1]}'

    glm_raw = smf.glm(formula=formula, data=df_fit, family=sm.families.Binomial())
    glm = glm_raw.fit()
    if save:
        glm.save(path)
    # print(fitted_model.summary())
    return glm


def rollout_ppo(environment, net, hp):
    batch_states = []
    batch_log_probs = []
    p_mean = []
    a_mean = []
    batch_rewards = []
    cov_mat = hp.cov_mat.to(hp.device)

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            _, policy_mean = net.forward(state)
            distro = MultivariateNormal(policy_mean, cov_mat)
            action = distro.sample().detach()
            log_prob = distro.log_prob(action).detach()

            # compute reward and go to next state
            r, state = environment.step(action.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_rewards.append(r)
            p_mean.append(policy_mean)
            a_mean.append(action)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_states = torch.stack(batch_states)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(hp.device)
    p_mean = torch.tensor(p_mean, dtype=torch.float).mean()
    a_mean = torch.tensor(a_mean, dtype=torch.float).mean()
    batch_returns = compute_returns(batch_rewards, hp.gamma).to(hp.device)

    return batch_states, batch_log_probs, batch_returns, p_mean, a_mean, torch.tensor(batch_rewards).mean()


# rollout for a2c
def rollout_a2c(environment, net, hp):
    batch_states = []
    batch_log_probs = []
    p_mean = []
    a_mean = []
    batch_rewards = []
    values = torch.tensor([]).to(hp.device)
    advantages = torch.tensor([]).to(hp.device)
    cov_mat = hp.cov_mat.to(hp.device)

    for _ in range(hp.batch_num):
        ep_rewards = []
        ep_values = torch.tensor([]).to(hp.device)
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            v, policy_mean = net.forward(state)
            distro = MultivariateNormal(policy_mean, cov_mat)
            action = distro.sample()
            log_prob = distro.log_prob(action)

            # compute reward and go to next state
            r, state = environment.step(action.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_values = torch.cat((ep_values, v))
            ep_rewards.append(r)
            p_mean.append(policy_mean)
            a_mean.append(action)
            batch_log_probs.append(log_prob)

        tgt, _ = net.forward(state)
        ep_adv = torch.zeros(len(ep_rewards)).to(hp.device)
        for t in reversed(range(len(ep_rewards))):
            tgt = ep_rewards[t] + hp.gamma * tgt
            adv = tgt - ep_values[t]
            ep_adv[t] = ((hp.gamma ** t) * adv).detach()
        values = torch.cat((values, ep_values))
        advantages = torch.cat((advantages, ep_adv))
        batch_rewards.append(ep_rewards)

    batch_states = torch.stack(batch_states)
    batch_log_probs = torch.stack(batch_log_probs)
    p_mean = torch.tensor(p_mean, dtype=torch.float).mean()
    a_mean = torch.tensor(a_mean, dtype=torch.float).mean()

    return batch_states, batch_log_probs, advantages, values, p_mean, a_mean, torch.tensor(batch_rewards).mean()


# rollout that requires gradient on policy log_prob
def rollout_reinforce(environment, net, hp):
    batch_log_probs = []
    p_mean = []
    a_mean = []
    batch_rewards = []
    cov_mat = hp.cov_mat.to(hp.device)

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            # compute action and log_prob
            policy_mean = net.forward(state)
            distro = MultivariateNormal(policy_mean, cov_mat)
            action = distro.sample()
            log_prob = distro.log_prob(action)

            # compute reward and go to next state
            r, state = environment.step(action.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_rewards.append(r)
            p_mean.append(policy_mean)
            a_mean.append(action)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_log_probs = torch.stack(batch_log_probs)
    p_mean = torch.tensor(p_mean, dtype=torch.float).mean()
    a_mean = torch.tensor(a_mean, dtype=torch.float).mean()
    batch_returns = compute_returns(batch_rewards, hp.gamma).to(hp.device)

    return batch_log_probs, batch_returns, p_mean, a_mean, torch.tensor(batch_rewards).mean()


def compute_returns(batch_rewards, gamma):
    batch_returns = []
    # iterate through each episode
    for ep_rewards in reversed(batch_rewards):
        discounted_reward = 0
        for r in reversed(ep_rewards):
            discounted_reward = r + discounted_reward * gamma
            batch_returns.insert(0, discounted_reward)

    return torch.tensor(batch_returns, dtype=torch.float)
