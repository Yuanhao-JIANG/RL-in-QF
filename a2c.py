import torch
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import statsmodels.api as sm
import env
import matplotlib.pyplot as plt


class LogLayer(nn.Module):
    def __init__(self):
        super(LogLayer, self).__init__()

    def forward(self, t):
        return torch.log(t+torch.abs(torch.min(t)).detach()+1)


# a2c network
class ActorCritic(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions

        # value
        self.critic_net = nn.Sequential(
            nn.Linear(num_state_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # policy
        self.actor_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            LogLayer(),     # to make numbers not differ too much
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_distro = self.actor_net(state)

        return value, policy_distro


# train method
def a2c(environment):
    # np.random.seed(123)
    # torch.manual_seed(211)

    learning_rate = 3e-2
    gamma = 0.99
    num_steps = 100
    max_episodes = 2000
    num_state_features = 18
    price_low = 400
    price_high = 2700
    price_step = 20
    num_actions = int((price_high - price_low)/price_step)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    actor_critic = ActorCritic(num_state_features, num_actions)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    actor_critic = actor_critic.to(device)
    actor_critic.train()

    plt.ion()

    for episode in range(max_episodes):
        state = torch.from_numpy(environment.reset()).to(device)
        I = 1
        total_reward = 0

        for steps in range(num_steps):
            value, policy_distro = actor_critic.forward(state)
            value, policy_distro = value.to(device)[0], policy_distro.to(device)

            action = torch.multinomial(policy_distro.cpu(), 1, replacement=True)[0]
            c = action * price_step + price_low
            log_prob = torch.log(policy_distro[action])

            # compute reward, go to next state to compute v'
            reward, new_state = environment.step(c.numpy())
            new_state = torch.from_numpy(new_state).to(device)
            value_next, _ = actor_critic.forward(new_state)
            value_next = value_next.to(device)[0]

            advantage = (reward + gamma * value_next - value).detach()
            critic_loss = - advantage * value
            actor_loss = - I * advantage * log_prob
            ac_loss = actor_loss + critic_loss

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

            I = I * gamma
            state = new_state
            total_reward += reward

        if episode % 5 == 0:
            sys.stdout.write("Episode: {}, avg_reward: {}\n".format(episode, total_reward/num_steps))
            plt.plot(episode, total_reward/num_steps, 'o')
            plt.pause(0.25)

    torch.save(actor_critic.state_dict(), './data/a2c_model.pth')


glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
a2c(env_train)
