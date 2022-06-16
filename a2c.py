import numpy as np
import pandas as pd
import torch
import sys
import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import statsmodels.api as sm
import env


# a2c network
class ActorCritic(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        # value
        self.critic_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # policy
        self.actor_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))  # [a,b,c,d] -> [[a,b,c,d]]

        value = self.critic_net(state)

        policy_dist = self.actor_net(state)
        policy_dist = F.softmax(policy_dist, dim=1)

        return value, policy_dist


def a2c(environment):
    learning_rate = 3e-4
    gamma = 0.99
    num_steps = 10
    max_episodes = 3000
    num_state_features = 4
    price_low = 400
    price_high = 2700
    num_actions = price_high - price_low + 1

    actor_critic = ActorCritic(num_state_features, num_actions)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        # go along a trajectory for num_steps steps
        state = environment.reset()
        for steps in range(num_steps):
            # we are using q values to approximate value
            value, policy_distro = actor_critic.forward(state)
            value = value[0][0]

            action = np.random.choice(num_actions, p=policy_distro.detach()[0].numpy())  # 0 or 1
            c = action + price_low
            log_prob = torch.log(policy_distro.squeeze(0)[action])    # log( P(action) )

            # go to next state
            reward, new_state = environment.step(c)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            state = new_state

            # record the q value for the last state
            q_val, _ = actor_critic.forward(new_state)
            q_val = q_val[0][0]

        if episode % 10 == 0:
            sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))

        # compute Q values
        q_vals = values.copy()
        for t in reversed(range(len(rewards))):
            # at start, use the last q_value to approximate the last value
            q_val = rewards[t] + gamma * q_val
            q_vals[t] = q_val

        # update actor critic
        values = torch.stack(values)
        q_vals = torch.stack(q_vals)
        log_probs = torch.stack(log_probs)

        # using q values to approximate values
        advantage = q_vals - values
        critic_loss = 0.5 * advantage.pow(2).mean()
        actor_loss = (-log_probs * advantage).mean()
        ac_loss = actor_loss + critic_loss

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        exit()


df_train = pd.read_csv('./data/dataframe_train.csv')
glm = sm.load('./data/glm.model')
env_train = env.Env(glm, df_train)
a2c(env_train)
