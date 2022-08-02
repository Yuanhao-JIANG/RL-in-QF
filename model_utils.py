import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


# actor network
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, int(out_dim)),
            nn.Threshold(.1, .1),
            # nn.Softmax(dim=0),
        )
        # self.net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())
        policy_distro = Categorical(self.net(state))
        return policy_distro


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
        # self.net.apply(init_weights)

    def forward(self, state):
        state = Variable(state.float())
        value = self.net(state)
        return value
