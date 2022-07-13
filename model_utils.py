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
            nn.Linear(128, num_actions),
            # take log to make numbers not differ too much with each other
            LogLayer(1),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.critic_net(state)
        policy_distro = self.actor_net(state)

        return value, policy_distro