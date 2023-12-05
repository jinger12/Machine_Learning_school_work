# global imports
import os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import scipy.stats as stats
import pandas as pd
import numpy as np
import sys

# local imports
from utils.attr_eda_utils import check_out_univariate_outliers_in_cap_x
from utils.attr_eda_utils import print_vifs


def get_outlier_residuals(a_series_of_residual, a_data_df):
    """

    :param a_series_of_residual:
    :param a_data_df:
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('get_outlier_residuals:')

    a_residuals_df = a_series_of_residual.to_frame()
    attr = a_residuals_df.columns[0]

    tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = \
        check_out_univariate_outliers_in_cap_x(a_residuals_df, [attr])

    outlier_residual_list = list(set(tukey_univariate_prob_outlier_dict[attr] +
                                     tukey_univariate_poss_outlier_dict[attr]))

    print(f'\noutlier_{attr}_list:', outlier_residual_list,  sep='')
    print(f'\na_data_df.loc[outlier_{attr}_list]:\n', a_data_df.loc[outlier_residual_list], sep='')

    return outlier_residual_list


def print_residual_vs_fitted_plot_guide():
    """

    :return:
    """
    print('\nThe residual vs fitted (or studentized residual vs fitted) plot is used to detect non-linearity, unequal '
          'error variances, and outliers.')
    print('\nHere are the characteristics of a well-behaved residual vs. fitted (or studentized residual vs fitted) '
          'plot and what they suggest about the appropriateness of the simple linear regression model:')
    print('\n*** non-linearity:')
    print('The residuals "bounce randomly" around the 0 line. This suggests that the assumption that the '
          'relationship is linear is reasonable.')
    print('\n*** unequal error variances:')
    print('The residuals roughly form a "horizontal band" around the 0 line. This suggests that the variances of the '
          'error terms are equal.')
    print('\n*** outliers:')
    print('No one residual "stands out" from the basic random pattern of residuals. This suggests that there are no '
          'outliers.')


def residuals_vs_fitted(fitted_sm_ols_model, data_df, studentized_residuals=False):
    """

    :param studentized_residuals:
    :param fitted_sm_ols_model:
    :param data_df: contains the data used to fit the model - may contain an id attribute as well
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('residuals_vs_fitted plot:')
    print_residual_vs_fitted_plot_guide()

    # set plotting params
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8, 7)

    # get the data from the fitted_sm_ols_model
    if studentized_residuals:
        residuals = OLSInfluence(fitted_sm_ols_model).resid_studentized
        residuals.name = 'studentized_residuals'
    else:
        residuals = fitted_sm_ols_model.resid
        residuals.name = 'residuals'

    fitted = fitted_sm_ols_model.fittedvalues
    smoothed = lowess(residuals, fitted)

    # get the outlier residuals
    outlier_residual_list = get_outlier_residuals(residuals, data_df)

    # plot residuals vs fitted and annotate outliers
    fig, ax = plt.subplots()
    ax.scatter(fitted, residuals, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_ylabel(residuals.name)
    ax.set_xlabel('fitted_values')
    ax.set_title(f'{residuals.name} vs. fitted_values')
    for i in outlier_residual_list:
        ax.annotate(i, xy=(fitted[i], residuals[i]))
    plt.grid()
    plt.show()


def qq_plot(fitted_sm_ols_model, data_df):
    """

    :param fitted_sm_ols_model:
    :param data_df:
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('qq_plot:')

    # set plotting params
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8, 7)

    # get the data from the fitted_sm_ols_model
    residuals = pd.Series(OLSInfluence(fitted_sm_ols_model).resid_studentized)
    residuals.name = 'studentized_residuals'

    # prepare the data for plotting
    sorted_residuals = residuals.sort_values(ascending=True)
    attr = 'sorted_' + str(sorted_residuals.name)
    sorted_residuals.name = attr
    sorted_residuals_df = sorted_residuals.to_frame()
    sorted_residuals_df['theoretical_quantiles'] = stats.probplot(sorted_residuals_df[attr], dist='norm', fit=False)[0]

    # get the outlier residuals
    outlier_residual_list = get_outlier_residuals(residuals, data_df)

    # plot the data
    fig, ax = plt.subplots()
    x = sorted_residuals_df['theoretical_quantiles']
    y = sorted_residuals_df[attr]
    ax.scatter(x, y, edgecolor='k', facecolor='none')
    ax.set_title('normal q-q')
    ax.set_ylabel(residuals.name)
    ax.set_xlabel('theoretical quantiles')
    ax.plot([np.min([x, y]), np.max([x, y])], [np.min([x, y]), np.max([x, y])], color='r', ls='--')
    for val in outlier_residual_list:
        ax.annotate(val, xy=(sorted_residuals_df['theoretical_quantiles'].loc[val], sorted_residuals_df[attr].loc[val]))
    plt.grid()
    plt.show()

    test_for_normality(fitted_sm_ols_model.resid)


def test_for_heteroscedasticity(fitted_sm_ols_model):
    """

    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('test_for_heteroscedasticity:')

    # https://medium.com/@remycanario17/tests-for-heteroskedasticity-in-python-208a0fdb04ab

    # set configuration
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

    # white heteroskedastic test
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_white.html
    print('\nwhite heteroskedastic test - two test statistics (LM and F): null hypothesis for both is that the model '
          'is homoskedastic', sep='')
    lm, lm_p_value, f_value, f_p_value = het_white(fitted_sm_ols_model.resid, fitted_sm_ols_model.model.exog)
    for label, value in zip(labels, [lm, lm_p_value, f_value, f_p_value]):
        print(f'   {label}: {value}')

    # breuschpagan heteroskedastic test
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
    print('\nbreuschpagan heteroskedastic test - two test statistics (LM and F): null hypothesis for both is that the '
          'model is homoskedastic', sep='')
    lm, lm_p_value, f_value, f_p_value = het_breuschpagan(fitted_sm_ols_model.resid, fitted_sm_ols_model.model.exog)
    for label, value in zip(labels, [lm, lm_p_value, f_value, f_p_value]):
        print(f'   {label}: {value}')


def scale_location_plot(fitted_sm_ols_model, data_df):
    """

    :param fitted_sm_ols_model:
    :param data_df:
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('scale_location_plot:')

    # prepare the data for plotting
    student_residuals = fitted_sm_ols_model.get_influence().resid_studentized_internal
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = fitted_sm_ols_model.resid.index
    fitted = fitted_sm_ols_model.fittedvalues
    smoothed = lowess(sqrt_student_residuals, fitted)  # locally weighted scatter plot smoothing

    # get the outlier residuals
    outlier_residual_list = get_outlier_residuals(sqrt_student_residuals, data_df)

    # plot the data
    fig, ax = plt.subplots()
    ax.scatter(fitted, sqrt_student_residuals, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_ylabel('sqrt(|Studentized Residuals|)')
    ax.set_xlabel('fitted values')
    ax.set_title('scale-location')
    ax.set_ylim(0, max(sqrt_student_residuals) + 0.1)
    for i in outlier_residual_list:
        ax.annotate(i, xy=(fitted[i], sqrt_student_residuals[i]))
    plt.grid()
    plt.show()


def residual_vs_leverage_plot(fitted_sm_ols_model, data_df):
    """
    
    :param fitted_sm_ols_model: 
    :param data_df: 
    :return: 
    """
    print('\n', 60 * '*', '\n', 60 * '*', sep='')
    print('residual_vs_leverage_plot:')
    
    # prepare the data
    student_residuals = pd.Series(fitted_sm_ols_model.get_influence().resid_studentized_internal)
    student_residuals.index = fitted_sm_ols_model.resid.index
    student_residuals.name = 'student_residuals'
    df = pd.DataFrame(student_residuals)
    df.columns = ['student_residuals']
    df['leverage'] = fitted_sm_ols_model.get_influence().hat_matrix_diag
    smoothed = lowess(df['student_residuals'], df['leverage'])

    # get the outlier residuals
    outlier_residual_list = get_outlier_residuals(student_residuals, data_df)

    # plot the data
    fig, ax = plt.subplots()
    x = df['leverage']
    y = df['student_residuals']
    x_pos = max(x) + max(x) * 0.01
    ax.scatter(x, y, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_ylabel('studentized residuals')
    ax.set_xlabel('leverage')
    ax.set_title('residuals vs. leverage')
    ax.set_ylim(min(y) - min(y) * 0.15, max(y) + max(y) * 0.15)
    ax.set_xlim(-0.01, max(x) + max(x) * 0.05)
    plt.tight_layout()
    for val in outlier_residual_list:
        ax.annotate(val, xy=(x.loc[val], y.loc[val]))

    cooks_x = np.linspace(min(x), x_pos, 50)
    p = len(fitted_sm_ols_model.params)
    pos_cooks_1y = np.sqrt((p * (1 - cooks_x)) / cooks_x)
    pos_cooks_05y = np.sqrt(0.5 * (p * (1 - cooks_x)) / cooks_x)
    neg_cooks_1y = -np.sqrt((p * (1 - cooks_x)) / cooks_x)
    neg_cooks_05y = -np.sqrt(0.5 * (p * (1 - cooks_x)) / cooks_x)

    ax.plot(cooks_x, pos_cooks_1y, label="Cook's Distance", ls=':', color='r')
    ax.plot(cooks_x, pos_cooks_05y, ls=':', color='r')
    ax.plot(cooks_x, neg_cooks_1y, ls=':', color='r')
    ax.plot(cooks_x, neg_cooks_05y, ls=':', color='r')
    ax.plot([0, 0], ax.get_ylim(), ls=":", alpha=.3, color='k')
    ax.plot(ax.get_xlim(), [0, 0], ls=":", alpha=.3, color='k')
    ax.annotate('1.0', xy=(x_pos, pos_cooks_1y[-1]), color='r')
    ax.annotate('0.5', xy=(x_pos, pos_cooks_05y[-1]), color='r')
    ax.annotate('1.0', xy=(x_pos, neg_cooks_1y[-1]), color='r')
    ax.annotate('0.5', xy=(x_pos, neg_cooks_05y[-1]), color='r')
    ax.legend()
    plt.grid()
    plt.show()


def plot_pred_vs_actual(fitted_sm_ols_model, y_series):
    """

    :param y_series: actual target values as a pandas series
    :param fitted_sm_ols_model:
    :return:
    """

    # prepare data for plotting
    x = y_series.array
    y = fitted_sm_ols_model.fittedvalues.array

    # plot the data
    fig, ax = plt.subplots()
    ax.scatter(x, y, edgecolors='k', facecolors='none')
    ax.plot([np.min([x, y]), np.max([x, y])], [np.min([x, y]), np.max([x, y])], color='r', ls='--')
    ax.set_ylabel('predicted')
    ax.set_xlabel('actual')
    ax.set_title('predicted vs. actual')
    plt.grid()
    plt.show()


def test_for_normality(a_series):
    """

    :param a_series:
    :return:
    """
    statistic, p_value = stats.normaltest(a_series)
    print('\ntest data for normality:', sep='')
    print(f'\nnull hypothesis: data comes from a normal distribution - p_value: {p_value}')


def plot_lin_reg_diagnostics(fitted_sm_ols_model, data_df, attrs_in_model, descriptive_attrs=None,
                             studentized_residuals=False, y_series=None):
    """

    :param y_series: actual target values as a pandas series
    :param descriptive_attrs:
    :param attrs_in_model:
    :param fitted_sm_ols_model:
    :param data_df:
    :param studentized_residuals:
    :return:
    """
    print('\n', 80 * '*', '\n', 80 * '*', '\n', 80 * '*', sep='')
    print('plot_lin_reg_diagnostics:')

    # https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a

    # set configuration
    if descriptive_attrs is None:
        descriptive_attrs = []

    _ = print_vifs(data_df, attrs_in_model)
    residuals_vs_fitted(fitted_sm_ols_model, data_df[attrs_in_model + descriptive_attrs],
                        studentized_residuals=studentized_residuals)
    test_for_heteroscedasticity(fitted_sm_ols_model)
    qq_plot(fitted_sm_ols_model, data_df[attrs_in_model + descriptive_attrs])
    scale_location_plot(fitted_sm_ols_model, data_df[attrs_in_model + descriptive_attrs])
    residual_vs_leverage_plot(fitted_sm_ols_model, data_df[attrs_in_model + descriptive_attrs])
    if y_series is not None:
        plot_pred_vs_actual(fitted_sm_ols_model, y_series)


if __name__ == '__main__':
    pass
