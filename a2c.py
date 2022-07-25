import torch
from torch.distributions import MultivariateNormal
import sys
import torch.optim as optim
import statsmodels.api as sm
from argparse import Namespace
import pandas as pd
import env
from model_utils import ActorCritic
from data_utils import rollout_r_p_a


# train method
def a2c(environment, hp):
    cov_mat = hp.cov_mat.to(hp.device)

    actor_critic = ActorCritic(hp.num_state_features, hp.price_min, hp.price_max)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=hp.lr)

    actor_critic = actor_critic.to(hp.device)

    # record model status before training
    mean_reward, p_mean, a_mean = rollout_r_p_a(environment, actor_critic, hp)
    moving_avg_reward = mean_reward
    moving_avg_reward_pool = [moving_avg_reward]
    sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(0, moving_avg_reward))
    sys.stdout.write("policy_mean: {}, action: {}\n".format(p_mean, a_mean))
    p_mean, a_mean = 0, 0

    actor_critic.train()
    for i in range(hp.num_itr):
        state = torch.from_numpy(environment.reset())
        state = torch.cat(
            (state[:-1], torch.tensor([state[-1] == 0, state[-1] == 1, state[-1] == 2], dtype=torch.float))
        ).to(hp.device)

        for step in range(hp.episode_size):
            value, policy_mean = actor_critic.forward(state)
            distro = MultivariateNormal(policy_mean, cov_mat)
            action = distro.sample()
            log_prob = distro.log_prob(action)

            # compute reward, go to next state to compute v'
            reward, new_state = environment.step(action.item())
            new_state = torch.from_numpy(new_state)
            new_state = torch.cat(
                (new_state[:-1],
                 torch.tensor([new_state[-1] == 0, new_state[-1] == 1, new_state[-1] == 2], dtype=torch.float))
            ).to(hp.device)
            value_next, _ = actor_critic.forward(new_state)

            advantage = (reward + hp.gamma * value_next[0] - value[0]).detach()
            critic_loss = - advantage * value[0]
            actor_loss = - (hp.gamma ** step) * advantage * log_prob
            ac_loss = actor_loss + critic_loss

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

            state = new_state
            moving_avg_reward += (reward - moving_avg_reward) / \
                                 (hp.batch_num * hp.episode_size + i * hp.episode_size + step + 1)
            p_mean += policy_mean.item()
            a_mean += action.item()

        itr = i + 1
        if itr % hp.batch_num == 0:
            sys.stdout.write("Iteration: {}, moving average reward: {}\n".format(itr, moving_avg_reward))
            sys.stdout.write("policy_mean: {}, action: {}\n".format(
                p_mean / (hp.batch_num * hp.episode_size), a_mean / (hp.batch_num * hp.episode_size))
            )
            moving_avg_reward_pool.append(moving_avg_reward)
            p_mean, a_mean = 0, 0

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False)
    torch.save(actor_critic.state_dict(), hp.model_save_path)


hyperparameter = Namespace(
    lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=5,
    episode_size=300,
    num_state_features=21,
    price_min=400,
    price_max=2700,
    cov_mat=torch.diag(torch.full(size=(1,), fill_value=50.)),
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    model_save_path='./data/a2c_model.pth',
    csv_out_path='./data/a2c_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
a2c(env_train, hyperparameter)
