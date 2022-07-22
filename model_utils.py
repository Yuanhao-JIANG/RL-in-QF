import torch
import torch.nn as nn
from torch.autograd import Variable


class Mul(nn.Module):
    def __init__(self, k):
        super(Mul, self).__init__()
        self.k = k

    def forward(self, t):
        return torch.mul(t, self.k)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# a2c network
class ActorCritic(nn.Module):
    def __init__(self, num_state_features, price_min, price_max):
        super(ActorCritic, self).__init__()
        self.price_min = price_min
        self.price_delta = price_max - price_min

        # value
        self.critic_net = nn.Sequential(
            nn.Linear(num_state_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # policy
        self.actor_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            Mul(1e-3),
            nn.Sigmoid()
        )
        self.actor_net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_mean = self.actor_net(state) * self.price_delta + self.price_min

        return value, policy_mean


class Reinforce(nn.Module):
    def __init__(self, num_state_features, price_min, price_max):
        super(Reinforce, self).__init__()
        self.price_min = price_min
        self.price_delta = price_max - price_min

        self.net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            Mul(1e-3),
            nn.Sigmoid()
        )
        self.net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())

        policy_mean = self.net(state) * self.price_delta + self.price_min

        return policy_mean


class PPO(nn.Module):
    def __init__(self, num_state_features, price_min, price_max):
        super(PPO, self).__init__()
        self.price_min = price_min
        self.price_delta = price_max - price_min

        # value
        self.critic_net = nn.Sequential(
            nn.Linear(num_state_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # policy
        self.actor_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            Mul(1e-3),
            nn.Sigmoid()
        )
        self.actor_net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_mean = self.actor_net(state) * self.price_delta + self.price_min

        return value, policy_mean
