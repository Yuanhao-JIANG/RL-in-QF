import torch
import torch.optim as optim
import statsmodels.api as sm
from torch.distributions import Categorical
from argparse import Namespace
import env
from model_utils import PPO


def ppo(environment, hp):
    # np.random.seed(123)
    # torch.manual_seed(211)

    num_actions = int((hp.price_high - hp.price_low) / hp.price_step)
    ppo_net = PPO(hp.num_state_features, num_actions)
    ppo_optimizer = optim.Adam(ppo_net.parameters(), lr=hp.lr)

    ppo_net = ppo_net.to(hp.device)
    ppo_net.train()

    moving_avg_reward = 0

    while True:
        batch_states, batch_actions, batch_log_probs, batch_returns = rollout(environment, ppo_net, hp)


def rollout(environment, net, hp):
    # Batch data. For more details, check function header.
    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []

    for _ in range(hp.batch_num):
        # rewards per episode
        ep_rewards = []
        state = torch.from_numpy(environment.reset())
        state = torch.cat(
            (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
        ).to(hp.device)

        # run an episode
        for _ in range(hp.episode_size):
            batch_states.append(state)

            # compute action and log_prob
            _, policy_distro = net.forward(state)
            distro = Categorical(policy_distro)
            action = distro.sample().detach()
            c = action.item() * hp.price_step + hp.price_low
            log_prob = distro.log_prob(action).detach()

            # compute reward and go to next state
            r, state = environment.step(c)
            state = torch.from_numpy(state)
            state = torch.cat(
                (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
            ).to(hp.device)

            ep_rewards.append(r)
            batch_actions.append(action)
            batch_log_probs.append(log_prob)

        batch_rewards.append(ep_rewards)

    batch_obs = torch.stack(batch_states)
    batch_acts = torch.tensor(batch_actions, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    batch_returns = compute_returns(batch_rewards, hp.gamma)

    return batch_obs, batch_acts, batch_log_probs, batch_returns


def compute_returns(batch_rewards, gamma):
    batch_returns = []
    # iterate through each episode
    for ep_rewards in reversed(batch_rewards):
        discounted_reward = 0
        for r in reversed(ep_rewards):
            discounted_reward = r + discounted_reward * gamma
            batch_returns.insert(0, discounted_reward)

    return torch.tensor(batch_returns, dtype=torch.float)


hyperparameter = Namespace(
    lr=3e-2,
    gamma=0.99,
    batch_num=10,
    episode_size=200,
    num_state_features=21,
    price_low=400,
    price_high=2700,
    price_step=20,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
ppo(env_train, hyperparameter)
