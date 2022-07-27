import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = [None, None, None]
df[0] = pd.read_csv('data/a2c_out.csv', header=None)
df[1] = pd.read_csv('data/reinforce_out.csv', header=None)
df[2] = pd.read_csv('data/ppo_out.csv', header=None)

fig, ax = plt.subplots(3)
fig.set_size_inches(11, 13)
fig.text(0.04, 0.5, 'moving average reward', va='center', rotation='vertical')
fig.text(0.5, 0.06, 'iteration', ha='center')
for idx in range(3):
    if idx == 0:
        ax[idx].set_title("moving average reward with a2c")
    elif idx == 1:
        ax[idx].set_title("moving average reward with reinforce")
    else:
        ax[idx].set_title("moving average reward with ppo")
    y = np.zeros((df[idx].shape[0], df[idx].shape[1]))
    x = np.arange(df[idx].shape[1])
    for i in range(6):
        y[i] = df[idx].to_numpy()[i]
        ax[idx].plot(x, y[i])
plt.show()
