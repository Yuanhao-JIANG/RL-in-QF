import torch
import sys
import torch.optim as optim
import statsmodels.api as sm
from argparse import Namespace
import pandas as pd
import env
from model_utils import Actor, Critic
from data_utils import rollout_a2c


# train method
def a2c(environment, hp):
    actor, critic = \
        Actor(hp.num_state_features, (hp.price_max - hp.price_min) / hp.price_binwidth), Critic(hp.num_state_features)
    actor_optimizer, critic_optimizer = \
        optim.Adam(actor.parameters(), lr=hp.actor_lr), optim.Adam(critic.parameters(), lr=hp.critic_lr)

    actor, critic = actor.to(hp.device), critic.to(hp.device)

    moving_avg_reward_pool = []

    actor.train()
    critic.train()
    for i in range(hp.num_itr):
        # ======================================= update network per batch ============================================
        # _, batch_log_probs, advantages, discounted_advantages, values, price_mean, moving_avg_reward = \
        #     rollout_a2c(environment, actor, critic, hp)
        #
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # discounted_advantages = \
        #     (discounted_advantages - discounted_advantages.mean()) / (discounted_advantages.std() + 1e-10)
        # critic_loss_mean = (- advantages * values).mean()
        # actor_loss_mean = (- discounted_advantages * batch_log_probs).mean()
        #
        # actor_optimizer.zero_grad()
        # actor_loss_mean.backward()
        # actor_optimizer.step()
        #
        # critic_optimizer.zero_grad()
        # critic_loss_mean.backward()
        # critic_optimizer.step()
        # =============================================================================================================

        # ======================================= update network per step =============================================
        moving_avg_reward = 0
        price_mean = 0
        actor_loss_mean = 0
        critic_loss_mean = 0
        state = torch.from_numpy(environment.reset()).to(hp.device)
        for step in range(hp.episode_size):
            policy_distro = actor.forward(state)
            action = policy_distro.sample()
            log_prob = policy_distro.log_prob(action)
            price = hp.price_min + action * hp.price_binwidth
            v = critic.forward(state)

            r, state = environment.step(price.item())
            state = torch.from_numpy(state).to(hp.device)
            v_next = critic.forward(state)
            advantage = (r + hp.gamma * v_next - v).detach()
            actor_loss = - (hp.gamma ** step) * advantage * log_prob
            critic_loss = - advantage * v
            actor_loss_mean += actor_loss
            critic_loss_mean += critic_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            moving_avg_reward += r
            price_mean += price

        critic_loss_mean = critic_loss_mean / hp.episode_size
        actor_loss_mean = actor_loss_mean / hp.episode_size
        moving_avg_reward = moving_avg_reward / hp.episode_size
        price_mean = price_mean / hp.episode_size
        # =============================================================================================================

        if i % 5 == 0:
            sys.stdout.write("Iteration: {}, price_mean: {}, moving average reward: {}\n"
                             .format(i, price_mean, moving_avg_reward))
            print(f'actor_loss: {actor_loss_mean.item()}, critic_loss: {critic_loss_mean.item()}')
            moving_avg_reward_pool.append(moving_avg_reward)

    # save training result to csv file, and save the model
    df = pd.DataFrame([moving_avg_reward_pool])
    df.to_csv(hp.csv_out_path, mode='a', header=False, index=False)
    torch.save(actor.state_dict(), hp.actor_save_path)
    torch.save(critic.state_dict(), hp.critic_save_path)


hyperparameter = Namespace(
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    num_itr=3000,
    batch_num=1,
    episode_size=300,
    moving_avg_num=300,
    num_state_features=21,
    price_min=200,
    price_max=2000,
    price_binwidth=15,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    actor_save_path='./data/a2c_actor.pth',
    critic_save_path='./data/a2c_critic.pth',
    csv_out_path='./data/a2c_out.csv'
)
glm = sm.load('./data/glm.model')
env_train = env.Env(glm)
a2c(env_train, hyperparameter)
# while True:
#     a2c(env_train, hyperparameter)
