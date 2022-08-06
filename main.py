from data_utils import generate_dataframe, fit_glm, plot_csv

# generate data and use that to fit glm
# df = generate_dataframe(data_size=10000, save=True, path='./data/dataframe_fit.csv', seed=0)
# glm = fit_glm(save=True)
# print(glm.summary())

# plot the result
# plot_csv(models=['a2c']*6, indices=range(6))
# plot_csv(models=['reinforce']*6, indices=range(6))
# plot_csv(models=['ppo']*6, indices=range(6))
# plot_csv(models=['a2c', 'reinforce', 'ppo'], indices=[5, 5, 5])
