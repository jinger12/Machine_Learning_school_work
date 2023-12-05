# global imports
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
# import sys

# local imports
import utils.attr_eda_utils as eda_utils
from utils.all_attr_eda_utils import check_out_target_imbalance, check_out_target_distribution


def bivariate_normal_test(x, y, alpha=0.05):
    """
    Null hypothesis: cap_x  comes from a multivariate normal distribution.
    :param y:
    :param x:
    :param alpha:
    :return:
    """

    # form a multivariate normal distribution
    cap_x = np.concatenate((x.values.reshape((-1, 1)), y.values.reshape((-1, 1))), axis=1)

    # conduct bivariate normality test
    hz, p_value, normal = pg.multivariate_normality(cap_x, alpha)

    return hz, p_value, normal


def get_correlation(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    # test to see if cap_x comes from a multivariate normal distribution
    _, _, normal = bivariate_normal_test(x, y, alpha=0.05)

    # apply correct method to derive the row and p
    if normal:
        method = 'pearson'
        row, p_value = stats.pearsonr(x.values, y.values)
    else:
        method = 'spearman'
        row, p_value = stats.spearmanr(x.values, y.values, nan_policy='omit')

    return row, p_value, method


def print_joint_plot_and_correlation(a_df, a_num_attr_list, a_target_attr):

    for attr in a_num_attr_list:

        print('\n', 20 * '*', sep='')
        print(attr)
        row, p_value, method = get_correlation(a_df[attr], a_df[a_target_attr])
        print(f'{method} correlation coefficient: {row} with p-value {p_value}')
        print('correlation coefficient p-value roughly indicates the probability of an uncorrelated system producing '
              'datasets that have a Spearman correlation at least as extreme as the one computed from these datasets.')
        sns.jointplot(data=a_df, x=attr, y=a_target_attr, kind="reg")
        plt.grid()
        plt.show()

        print(f'\nstatsmodels OLS - simple linear regression - regress {a_target_attr} on {attr}')
        temp_df = a_df.copy()
        print(f'before drop nans in a_df - a_df.isna().sum().sum(): {temp_df.isna().sum().sum()}')
        temp_df = temp_df.dropna()
        print(f'before drop nans in a_df - a_df.isna().sum().sum(): {temp_df.isna().sum().sum()}')
        cap_x = sm.add_constant(temp_df[attr])
        est = sm.OLS(temp_df[a_target_attr], cap_x).fit()
        print('statsmodels OLS summary()\n')
        print(est.summary())


def numerical_attr_eda_regression(a_df, a_num_attr_list, a_target_attr):
    """

    :param a_target_attr:
    :param a_df:
    :param a_num_attr_list:
    :return:
    """

    print('\n', 60 * '*', sep='')
    print('numerical_attr_eda_regression - bivariate analysis: target attribute vs numerical attributes')
    print_joint_plot_and_correlation(a_df, a_num_attr_list, a_target_attr)


def numerical_attr_eda_classification(a_df, a_num_attr_list, a_target_attr=None):
    """

    :param a_target_attr:
    :param a_df:
    :param a_num_attr_list:
    :return:
    """

    print('\n', 60 * '*', sep='')
    print('numerical_attr_eda_classification - bivariate analysis: target attribute vs categorical attributes')
    for attr in a_df[a_num_attr_list].columns:
        pivot_a_df = a_df.pivot(columns=a_target_attr, values=attr)
        sns.catplot(kind='box', data=pivot_a_df)
        plt.ylabel(attr)
        plt.grid()
        plt.show()


def numerical_attr_eda(a_df, a_num_attr_list, a_prediction_task_type, a_target_attr, show_outliers=False):
    """

    :param a_target_attr:
    :param a_df:
    :param a_num_attr_list:
    :param a_prediction_task_type:
    :param show_outliers:
    :return:
    """
    print('\n', 80 * '*', sep='')
    print('numerical_attr_eda')
    print('a_prediction_task_type:', a_prediction_task_type, '\n')

    if len(a_num_attr_list) == 0:
        print(f'\nthere are no numerical attributes')
        return None

    eda_utils.common_numerical_attr_eda_tasks(a_df, a_num_attr_list, show_outliers=show_outliers)

    if a_prediction_task_type == 'regression':
        check_out_target_distribution(a_df, a_target_attr)
        numerical_attr_eda_regression(a_df, a_num_attr_list, a_target_attr)
    elif a_prediction_task_type == 'classification':
        check_out_target_imbalance(a_df, a_target_attr)
        numerical_attr_eda_classification(a_df, a_num_attr_list, a_target_attr)
    else:
        raise TypeError('unrecognized prediction_type_task!!!')


if __name__ == '__main__':
    pass
