import torch
import torch.nn as nn
from torch.autograd import Variable


class LogLayer(nn.Module):
    # k > 0, which is used to make sure the number we take log on is greater than 0
    def __init__(self, k):
        super(LogLayer, self).__init__()
        self.k = k

    def forward(self, t):
        return torch.log(t + torch.abs(torch.min(t)).detach() + self.k)


class Mul(nn.Module):
    def __init__(self, k):
        super(Mul, self).__init__()
        self.k = k

    def forward(self, t):
        return torch.mul(t, self.k)


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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            Mul(1e-3),
            nn.Softsign()
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_mean = (self.actor_net(state) + 1) * (self.price_delta / 2) + self.price_min

        return value, policy_mean


class Reinforce(nn.Module):
    def __init__(self, num_state_features):
        super(Reinforce, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        state = Variable(state.float())

        policy_mean = self.net(state)

        return policy_mean


class PPO(nn.Module):
    def __init__(self, num_state_features):
        super(PPO, self).__init__()

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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_mean = self.actor_net(state)

        return value, policy_mean
