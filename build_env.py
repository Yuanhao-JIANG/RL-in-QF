import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_glm(fit_df_path='./data/dataframe_fit.csv', save=False, path='./data/glm.model'):
    df_fit = pd.read_csv(fit_df_path)
    formula = 'response ~ gender'
    for i in range(len(df_fit.columns) - 2):
        formula += f' + {df_fit.columns[i + 1]}'

    glm_raw = smf.glm(formula=formula, data=df_fit, family=sm.families.Binomial())
    glm = glm_raw.fit()
    if save:
        glm.save(path)
    # print(fitted_model.summary())
    return glm


def test_glm(glm_path='./data/glm.model', test_df_path='./data/dataframe_test.csv', index=0, num=10):
    glm = sm.load(glm_path)
    df_test = pd.read_csv(test_df_path)
    instance = df_test.iloc[index:index + num]
    pred = glm.predict(instance)
    df_pred = pd.DataFrame(pred, columns=['pred'])
    df_pred['true'] = instance['response']
    print(df_pred)
