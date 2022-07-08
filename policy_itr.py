import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import statsmodels.api as sm
import env


class Net(nn.Module):
    def __init__(self, num_state_features, num_actions):
        super(Net, self).__init__()
        self.num_actions = num_actions

        # value
        self.value_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # policy
        self.policy_net = nn.Sequential(
            nn.Linear(num_state_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        state = Variable(state.float())

        value = self.value_net(state)
        policy_distro = self.policy_net(state)

        with torch.no_grad():
            v = self.value_net(state)

        return value, policy_distro, v


# train method
def policy_itr(environment):
    np.random.seed(123)
    torch.manual_seed(211)

    learning_rate = 3e-2
    gamma = 0.99
    num_steps = 100
    max_episodes = 2000
    num_state_features = 18
    price_low = 400
    price_high = 2700
    num_actions = price_high - price_low + 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Net(num_state_features, num_actions)
    net_optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net = net.to(device)
    net.train()

    for episode in range(max_episodes):
        state = torch.from_numpy(environment.reset()).to(device)
        total_reward = 0

        for steps in range(num_steps):
            # policy evaluation
            value, policy_distro, _ = net.forward(state)
            value, policy_distro = value.to(device)[0], policy_distro.to(device)
            c = torch.multinomial(policy_distro.cpu(), 1, replacement=True)[0] + price_low

            target = torch.tensor(0.).to(device)
            for i in range(len(policy_distro)):
                if policy_distro[i] > 0.02:
                    r, s_expect = environment.peek_expected(i + price_low)
                    _, _, v = net.forward(torch.from_numpy(s_expect).to(device))
                    target += policy_distro[i].detach() * (r + gamma * v[0])

            loss = (target - value).pow(2)

            if steps % 10 == 0:
                # policy improvement
                q = []
                for i in range(len(policy_distro)):
                    if policy_distro[i] > 0.02:
                        r, s_expect = environment.peek_expected(i + price_low)
                        _, _, v = net.forward(torch.from_numpy(s_expect).to(device))
                        q.append(r + gamma * v[0].cpu())
                idx_max = np.argmax(np.array(q))
                loss -= policy_distro[idx_max]

            net_optimizer.zero_grad()
            loss.backward()
            net_optimizer.step()

            # compute reward, and go to next state
            reward, state = environment.step(c.numpy())
            state = torch.from_numpy(state).to(device)
            total_reward += reward

        if episode % 10 == 0:
            sys.stdout.write("episode: {}, avg_reward: {}\n".format(episode, total_reward/num_steps))

    torch.save(net.state_dict(), './data/policy_itr__model.pth')


glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
policy_itr(env_train)
