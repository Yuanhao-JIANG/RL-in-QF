import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch

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
    formula = 'response ~ gender'
    for i in range(len(df_fit.columns) - 2):
        formula += f' + {df_fit.columns[i + 1]}'

    glm_raw = smf.glm(formula=formula, data=df_fit, family=sm.families.Binomial())
    glm_fit = glm_raw.fit()
    if save:
        glm_fit.save(path)
    return glm_fit


# df = generate_dataframe(data_size=10000, save=True, path='./data/dataframe_fit.csv', seed=0)
# glm = fit_glm(save=True)
# print(glm.summary())


def rollout_ppo(environment, actor, hp):
    batch_states = []
    batch_log_probs = []
    price_mean = []
    batch_rewards = []

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            policy_distro = actor.forward(state)
            action = policy_distro.sample()
            log_prob = policy_distro.log_prob(action).detach()
            price = hp.price_min + action * hp.price_binwidth

            # compute reward and go to next state
            r, state = environment.step(price.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_rewards.append(r)
            price_mean.append(price)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_states = torch.stack(batch_states)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(hp.device)
    price_mean = torch.tensor(price_mean, dtype=torch.float).mean()
    batch_returns, _ = compute_returns(batch_rewards, hp.gamma)

    return batch_states, batch_log_probs, batch_returns.to(hp.device), price_mean, \
        torch.flatten(torch.tensor(batch_rewards))[-hp.moving_avg_num:].mean()


# rollout for a2c
def rollout_a2c(environment, actor, critic, hp):
    batch_states = []
    batch_log_probs = []
    price_mean = []
    batch_rewards = []
    values = torch.tensor([]).to(hp.device)
    advantages = torch.tensor([]).to(hp.device)
    discounted_advantages = torch.tensor([]).to(hp.device)

    for _ in range(hp.batch_num):
        ep_rewards = []
        ep_values = torch.tensor([]).to(hp.device)
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            policy_distro = actor.forward(state)
            action = policy_distro.sample()
            log_prob = policy_distro.log_prob(action)
            price = hp.price_min + action * hp.price_binwidth
            v = critic.forward(state)

            # compute reward and go to next state
            r, state = environment.step(price.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_values = torch.cat((ep_values, v))
            ep_rewards.append(r)
            price_mean.append(price)
            batch_log_probs.append(log_prob)
        tgt = critic.forward(state)
        ep_adv = torch.zeros(len(ep_rewards)).to(hp.device)
        discounted_ep_adv = torch.zeros(len(ep_rewards)).to(hp.device)
        for t in reversed(range(len(ep_rewards))):
            tgt = ep_rewards[t] + hp.gamma * tgt
            adv = tgt - ep_values[t]
            ep_adv[t] = adv.detach()
            discounted_ep_adv[t] = ((hp.gamma ** t) * adv).detach()
        values = torch.cat((values, ep_values))
        advantages = torch.cat((advantages, ep_adv))
        discounted_advantages = torch.cat((discounted_advantages, discounted_ep_adv))
        batch_rewards.append(ep_rewards)

    batch_states = torch.stack(batch_states)
    batch_log_probs = torch.stack(batch_log_probs)
    price_mean = torch.tensor(price_mean, dtype=torch.float).mean()

    return batch_states, batch_log_probs, advantages, discounted_advantages, values, price_mean, \
        torch.flatten(torch.tensor(batch_rewards))[-hp.moving_avg_num:].mean()


# rollout that requires gradient on policy log_prob
def rollout_reinforce(environment, actor, hp):
    batch_log_probs = []
    price_mean = []
    batch_rewards = []

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset()).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            # compute action and log_prob
            policy_distro = actor.forward(state)
            action = policy_distro.sample()
            log_prob = policy_distro.log_prob(action)
            price = hp.price_min + action * hp.price_binwidth

            # compute reward and go to next state
            r, state = environment.step(price.item())
            state = torch.from_numpy(state).to(hp.device)

            ep_rewards.append(r)
            price_mean.append(price)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_log_probs = torch.stack(batch_log_probs)
    price_mean = torch.tensor(price_mean, dtype=torch.float).mean()
    _, discounted_returns = compute_returns(batch_rewards, hp.gamma)

    return batch_log_probs, discounted_returns.to(hp.device), price_mean, \
        torch.flatten(torch.tensor(batch_rewards))[-hp.moving_avg_num:].mean()


def compute_returns(batch_rewards, gamma):
    batch_returns = []
    discounted_batch_returns = []
    # iterate through each episode
    for ep_rewards in reversed(batch_rewards):
        discounted_reward = 0
        for t in reversed(range(len(ep_rewards))):
            discounted_reward = ep_rewards[t] + discounted_reward * gamma
            batch_returns.insert(0, discounted_reward)
            discounted_batch_returns.insert(0, (gamma ** t) * discounted_reward)

    return torch.tensor(batch_returns, dtype=torch.float), torch.tensor(discounted_batch_returns, dtype=torch.float)
