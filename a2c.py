import numpy as np
import pandas as pd
import torch
import sys
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
        state = Variable(state.float().unsqueeze(0))  # [a,b,c,d] -> [[a,b,c,d]]

        value = self.critic_net(state)

        policy_dist = self.actor_net(state)
        policy_dist = F.softmax(policy_dist, dim=1)

        return value, policy_dist


# train method
def a2c(environment):
    learning_rate = 3e-4
    gamma = 0.99
    num_steps = 50
    max_episodes = 2000
    num_state_features = 4
    price_low = 400
    price_high = 2700
    num_actions = price_high - price_low + 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    actor_critic = ActorCritic(num_state_features, num_actions)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    actor_critic = actor_critic.to(device)
    actor_critic.train()
    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        # go along a trajectory for num_steps steps
        state = environment.reset()
        for steps in range(num_steps):
            state = state.to(device)
            value, policy_distro = actor_critic.forward(state)
            value, policy_distro = value.to(device), policy_distro.to(device)
            value = value[0][0]

            action = np.random.choice(num_actions, p=policy_distro.detach().cpu()[0].numpy())  # 0 or 1
            c = action + price_low
            log_prob = torch.log(policy_distro.squeeze(0)[action])    # log( P(action) )

            # go to next state
            reward, new_state = environment.step(c)
            new_state = new_state.to(device)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            state = new_state

            if episode % 10 == 0 and steps == num_steps - 1:
                # record the value for the last state to compute the last q value
                val_last, _ = actor_critic.forward(new_state)
                val_last = val_last[0][0]

                sys.stdout.write("episode: {}, reward: {}, state: {}\n".format(episode, np.mean(rewards),
                                                                               state.detach().cpu().numpy()))

        # compute Q values
        q_vals = values.copy()
        q_vals[-1] = rewards[-1] + gamma * val_last
        for t in reversed(range(len(rewards) - 1)):
            q_vals[t] = rewards[t] + gamma * values[t + 1]

        # update actor critic
        values = torch.stack(values)
        q_vals = torch.stack(q_vals)
        log_probs = torch.stack(log_probs)

        # using q values to approximate values
        advantage = q_vals - values
        critic_loss = 0.5 * (values - q_vals.detach()).pow(2).mean()
        actor_loss = (-advantage.detach() * log_probs).mean()
        ac_loss = actor_loss + critic_loss

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    torch.save(actor_critic.state_dict(), './data/a2c_model.pth')


df_train = pd.read_csv('./data/dataframe_train.csv')
glm = sm.load('./data/glm.model')
env_train = env.Env(glm, df_train)
a2c(env_train)
