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
        m.bias.data.fill_(0.)


# actor network
class Actor(nn.Module):
    def __init__(self, in_dim, price_min, price_max):
        super(Actor, self).__init__()
        self.price_min = price_min
        self.price_delta = price_max - price_min
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
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


# critic network
class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())
        value = self.net(state)
        return value
