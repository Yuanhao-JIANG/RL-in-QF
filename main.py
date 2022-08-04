from data_utils import generate_dataframe, fit_glm, plot_csv
from a2c import train_a2c
from reinforce import train_reinforce
from ppo import train_ppo

# generate data and use that to fit glm
# df = generate_dataframe(data_size=10000, save=True, path='./data/dataframe_fit.csv', seed=0)
# glm = fit_glm(save=True)
# print(glm.summary())

# train different models
# train_a2c()
# train_reinforce()
# train_ppo()

# plot the result
# plot_csv(models=['a2c', 'reinforce', 'ppo'], indices=[0, 0, 0])
