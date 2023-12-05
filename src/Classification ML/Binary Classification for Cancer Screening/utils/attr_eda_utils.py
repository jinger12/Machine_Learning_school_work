# global imports
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import scipy as sp
from scipy.stats import chi2
import numpy as np
import statsmodels.api as sm
import time
from sklearn.preprocessing import StandardScaler
import sys

# local imports


def common_eda_tasks(a_df, an_attr_list):
    """

    :param a_df:
    :param an_attr_list:
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', '\n', 60 * '*', sep='')
    print('common_eda_tasks\n')

    print('an_attr_list:', an_attr_list)
    print('\na_df[an_attr_list].shape:', a_df[an_attr_list].shape)
    print('\na_df[an_attr_list].info():')
    a_df[an_attr_list].info()
    print('\na_df[an_attr_list].head():\n', a_df[an_attr_list].head())
    print('\nNA (np.nan or None) count - a_df[an_attr_list].isna().sum():\n',
          a_df[an_attr_list].isna().sum(), sep='')
    print('\nNA (np.nan or None) fraction - a_df[an_attr_list].isna().sum() / a_df.shape[0]:\n',
          a_df[an_attr_list].isna().sum() / a_df.shape[0], sep='')


def print_hist_of_num_attrs(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print('histograms of the numerical attributes:')
    a_df[a_num_attr_list].hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()


def print_pair_plot(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print('investigate multi co-linearity: pair plots of the numerical attributes:\n')
    if a_df[a_num_attr_list].shape[1] < 20:
        sns.pairplot(a_df[a_num_attr_list], height=1)
        plt.tight_layout()
        plt.show()
    else:
        print(f'\nSkip pair plots - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful visual '
              f'output.', sep='')


def check_out_multi_co_linearity_in_cap_x(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """
    print('\n', 60 * '*', '\n', 60 * '*', '\n', 60 * '*', sep='')
    print('check_out_multi_co_linearity_in_cap_x:')
    print_corr_of_num_attrs(a_df, a_num_attr_list)
    _ = print_vifs(a_df, a_num_attr_list)
    print_pair_plot(a_df, a_num_attr_list)


def get_flattened_corr_matrix(corr_df, corr_threshold=0.75):

    corr_df = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool_))
    flat_attr_correlations_df = corr_df.stack().reset_index()
    flat_attr_correlations_df = flat_attr_correlations_df.rename(columns={'level_0': 'attribute_x',
                                                                          'level_1': 'attribute_y',
                                                                          0: 'correlation'})
    flat_attr_correlations_df = \
        flat_attr_correlations_df[(flat_attr_correlations_df.correlation != 1) &
                                  (flat_attr_correlations_df.correlation.abs() > corr_threshold)].\
        sort_values('correlation').reset_index(drop=True)

    print('\n', f'{flat_attr_correlations_df.shape[0]} pairs of attributes with the absolute values of '
                f'correlations row > {corr_threshold}', sep='')
    print(flat_attr_correlations_df)

    return flat_attr_correlations_df


def get_correlation_data_frame(a_df, a_num_attr_list, method='pearson', corr_threshold=0.75):
    """

    :param a_df:
    :param a_num_attr_list:
    :param method:
    :param corr_threshold:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print(f'\ncorrelation data frame using {method} method with threshold {corr_threshold}:\n')

    attr_correlations_df = a_df[a_num_attr_list].corr(method=method)
    flat_attr_correlations_df = get_flattened_corr_matrix(attr_correlations_df, corr_threshold=corr_threshold)

    return flat_attr_correlations_df


def print_corr_of_num_attrs(a_df, a_num_attr_list, method='pearson', corr_threshold=0.25):
    """

    :param method:
    :param corr_threshold:
    :param a_df:
    :param a_num_attr_list:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print('heatmap of design matrix attribute correlations:\n')

    if a_df[a_num_attr_list].shape[1] < 20:
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(a_df[a_num_attr_list].corr(method=method), annot=True, ax=ax)
        plt.show()
        _ = get_correlation_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)
    else:
        print(f'\nSkip correlation heat map - {a_df[a_num_attr_list].shape[1]} attributes is too many for a useful '
              f'visual output.', sep='')
        _ = get_correlation_data_frame(a_df, a_num_attr_list, method=method, corr_threshold=corr_threshold)


def drop_obs_with_nans(a_df):
    """

    :param a_df:
    :return:
    """

    if a_df.isna().sum().sum() > 0:
        print(f'\nfound observations with nans - pre obs. drop a_df.shape: {a_df.shape}')
        a_df = a_df.dropna(axis=0, how='any')
        print(f'post obs. drop a_df.shape: {a_df.shape}')

    return a_df


def prep_data_for_vif_calc(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """

    # drop observations with nans
    a_df = drop_obs_with_nans(a_df[a_num_attr_list])

    # prepare the data - make sure you perform the analysis on the design matrix
    design_matrix = None
    bias_attr = None
    for attr in a_df[a_num_attr_list]:
        if a_df[attr].nunique() == 1 and a_df[attr].iloc[0] == 1:  # found the bias attribute
            design_matrix = a_df[a_num_attr_list]
            bias_attr = attr
            print('found the bias term - no need to add one')
            break

    if design_matrix is None:
        design_matrix = sm.add_constant(a_df[a_num_attr_list])
        bias_attr = 'const'
        print('\nAdded a bias term to the data frame to construct the design matrix for assessment of vifs.', sep='')

    # if numerical attributes in the data frame are not scaled then scale them - don't scale the bias term
    if not (a_df[a_num_attr_list].mean() <= 1e-14).all():
        print('scale the attributes - but not the bias term')
        design_matrix[a_num_attr_list] = StandardScaler().fit_transform(design_matrix[a_num_attr_list])

    return design_matrix, bias_attr


def print_vifs(a_df, a_num_attr_list, vif_inspection_threshold=2, ols_large_vifs=True):
    """

    :param a_df:
    :param a_num_attr_list:
    :param vif_inspection_threshold:
    :param ols_large_vifs:
    :return:
    """

    # VIF determines the strength of the correlation between the independent variables. It is predicted by taking a
    # variable and regressing it against every other variable.
    # VIF score of an independent variable represents how well the variable is explained by other independent variables.
    # https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/
    # https://www.statsmodels.org/v0.13.0/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

    print('\n', 40 * '*', sep='')
    print('investigate multi co-linearity - calculate variance inflation factors:')

    design_matrix, bias_attr = prep_data_for_vif_calc(a_df, a_num_attr_list)

    # calculate the vifs
    vif_df = pd.DataFrame()
    vif_df['attribute'] = design_matrix.columns.tolist()
    vif_df['vif'] = [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]
    vif_df['vif'] = vif_df['vif'].round(2)
    vif_df = vif_df.sort_values('vif')

    print('\n', vif_df, sep='')
    time.sleep(2)

    if (vif_df.vif.values > vif_inspection_threshold).any() and ols_large_vifs:
        check_out_large_vifs(vif_df, design_matrix, vif_inspection_threshold=vif_inspection_threshold)

    return vif_df


def check_out_large_vifs(vif_df, design_matrix, vif_inspection_threshold=2):

    vif_gt_threshold_list = vif_df.loc[vif_df.vif > vif_inspection_threshold, 'attribute'].tolist()

    if len(vif_gt_threshold_list) > 0:
        print(f'\nthe attributes {vif_gt_threshold_list} have vif values greater than {vif_inspection_threshold} - '
              f'let\'s look at the details of regressing them on the rest of the design matrix')

        for inf_vif_attr in sorted(vif_gt_threshold_list, reverse=True):
            predictors = design_matrix.columns.tolist()
            predictors.remove(inf_vif_attr)
            est = sm.OLS(design_matrix[inf_vif_attr], design_matrix[predictors]).fit()
            print(f'\n\n\n\n{inf_vif_attr}')
            print(est.summary())
    else:
        print(f'\nno attributes have vifs greater than {vif_inspection_threshold} - no further vif analysis required')


def print_boxplots_of_num_attrs(a_df, a_num_attr_list, tukey_outliers=False, show_outliers=False):
    """

    :param tukey_outliers:
    :param a_df:
    :param a_num_attr_list:
    :param show_outliers:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print('boxplots of the numerical attributes:\n')

    tukey_univariate_poss_outlier_dict = {}
    tukey_univariate_prob_outlier_dict = {}
    for attr in a_num_attr_list:
        print('\n', 20 * '*', sep='')
        print(attr, sep='')
        a_df.boxplot(column=attr, figsize=(5, 5))
        plt.show()
        if tukey_outliers:
            outliers_prob, outliers_poss = tukeys_method(a_df, attr)
            tukey_univariate_prob_outlier_dict[attr] = outliers_prob
            tukey_univariate_poss_outlier_dict[attr] = outliers_poss
            if show_outliers:
                print('univariate outliers:')
                print('\ntukey\'s method - outliers_prob indices:\n', outliers_prob, sep='')
                print('\ntukey\'s method - outliers_poss indices:\n', outliers_poss, sep='')

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def tukeys_method(a_df, variable):
    """
    # Takes two parameters: dataframe & variable of interest as string

    :param a_df:
    :param variable:
    :return:
    """

    q1 = a_df[variable].quantile(0.25)
    q3 = a_df[variable].quantile(0.75)
    iqr = q3 - q1
    inner_fence = 1.5 * iqr
    outer_fence = 3 * iqr

    # inner fence lower and upper end
    inner_fence_le = q1 - inner_fence
    inner_fence_ue = q3 + inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    outliers_prob = []
    outliers_poss = []
    for index, x in zip(a_df.index, a_df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in zip(a_df.index, a_df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)

    return outliers_prob, outliers_poss


def use_tukeys_method(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """
    print('\n', 40 * '*', sep='')
    print('use_tukeys_method to identify outliers:\n')

    tukey_univariate_poss_outlier_dict = {}
    tukey_univariate_prob_outlier_dict = {}
    for attr in a_num_attr_list:
        print('\n', attr, sep='')
        outliers_prob, outliers_poss = tukeys_method(a_df, attr)
        print('tukey\'s method - outliers_prob indices: ', outliers_prob)
        tukey_univariate_prob_outlier_dict[attr] = outliers_prob
        print('tukey\'s method - outliers_poss indices: ', outliers_poss)
        tukey_univariate_poss_outlier_dict[attr] = outliers_poss

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def check_out_univariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    """

    :param a_df:
    :param a_num_attr_list:
    :param show_outliers:
    :return:
    """
    # https://journals.sagepub.com/doi/pdf/10.1177/0844562118786647
    # https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755

    print('\n', 40 * '*', sep='')
    print('check_out_univariate_outliers_in_cap_x:')

    print_hist_of_num_attrs(a_df, a_num_attr_list)

    tukey_outliers = True
    tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = \
        print_boxplots_of_num_attrs(a_df, a_num_attr_list, tukey_outliers=tukey_outliers, show_outliers=show_outliers)

    if not tukey_outliers:
        tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict = \
            use_tukeys_method(a_df, a_num_attr_list)

    if show_outliers:
        print('\ntukey_univariate_prob_outlier_dict:')
    attrs_with_tukey_prob_outliers_list = []
    univariate_outlier_list = []
    for attr, outliers_prob in tukey_univariate_prob_outlier_dict.items():
        if show_outliers:
            print('\n   attr:', attr, '; outliers_prob:', outliers_prob, sep='')
        if len(outliers_prob) > 0:
            attrs_with_tukey_prob_outliers_list.append(attr)
            univariate_outlier_list.extend(outliers_prob)

    print('\n', 30 * '*', sep='')
    print('univariate outlier summary:')
    print(f'\ncount of attributes with probable tukey univariate outliers:\n{len(attrs_with_tukey_prob_outliers_list)}')
    print(f'\nlist of attributes with probable tukey univariate outliers:\n{attrs_with_tukey_prob_outliers_list}')
    print(f'\ncount of unique probable tukey univariate outliers across all attributes:\n'
          f'{len(set(univariate_outlier_list))}')
    if show_outliers:
        print(f'\nlist of observations with probable tukey univariate outliers:\n{set(univariate_outlier_list)}')

    return tukey_univariate_poss_outlier_dict, tukey_univariate_prob_outlier_dict


def mahalanobis_method(a_df):
    """
    
    :param a_df: 
    :return: 
    """
    # https://www.youtube.com/watch?v=3IdvoI8O9hU
    # https://www.youtube.com/watch?v=spNpfmWZBmg

    # drop observations with nans
    a_df = drop_obs_with_nans(a_df)

    # calculate the mahalanobis distance
    x_minus_mu = a_df - np.mean(a_df)
    cov = np.cov(a_df.values.T)  # Covariance

    try:
        inv_covmat = sp.linalg.inv(cov)  # Inverse covariance
    except np.linalg.LinAlgError as e:
        print('\n', e)
        print(f'\nnumerical matrix is singular so mahalanobis_method threw an exception - multivariate outliers '
              f'analysis could not be completed')
        return None, None

    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # calculate threshold
    threshold = np.sqrt(chi2.ppf((1 - 0.001), df=a_df.shape[1]))  # degrees of freedom = number of variables

    # collect outliers
    outlier = []
    for index, value in enumerate(md):
        if value > threshold:
            outlier.append(index)
        else:
            continue

    return outlier, md


def check_out_multivariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    """

    :param a_df:
    :param a_num_attr_list:
    :param show_outliers:
    :return:
    """
    # https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
    print('\n', 40 * '*', sep='')
    print('check_out_multivariate_outliers_in_cap_x:')
    outlier, _ = mahalanobis_method(a_df[a_num_attr_list])

    print('\nmultivariate outlier summary:\n')

    if outlier is not None:
        print(f'count of multivariate outliers using mahalanobis method: {len(outlier)}')
        if show_outliers:
            print('\nmultivariate outliers using mahalanobis method:', outlier)


def check_out_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=False):
    """

    :param a_df:
    :param a_num_attr_list:
    :param show_outliers:
    :return:
    """

    print('\n', 60 * '*', '\n', 60 * '*', '\n', 60 * '*', sep='')
    print('check_out_outliers_in_cap_x:')
    _, _ = check_out_univariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=show_outliers)
    check_out_multivariate_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=show_outliers)

    # once we are done detecting outliers here is a reference on how to deal with them in model development
    # https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-3-dcb54abaf7b0


def explore_cardinality_of_categorical_attrs(a_df, a_cat_attr_list):
    """

    :param a_df:
    :param a_cat_attr_list:
    :return:
    """

    print('\n', 60 * '*', '\n', 60 * '*', '\n', 60 * '*', sep='')
    print('explore_cardinality_of_categorical_attrs:')

    for attr in a_cat_attr_list:
        print('\n', 20 * '*', sep='')
        print(attr)
        print('a_df[attr].nunique():', a_df[attr].nunique())
        print('a_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False), sep='')


def check_out_skew_and_kurtosis(a_df):
    print('\ncheck out skewness and kurtosis:')
    for attr in a_df.columns:
        print('\nattr: ', attr, sep='')
        print(f'kurtosis: {a_df[attr].kurtosis()}')
        print(f'skewness: {a_df[attr].skew()}')


def common_numerical_attr_eda_tasks(a_df, a_num_attr_list, show_outliers=False):
    """

    :param a_df:
    :param a_num_attr_list:
    :param show_outliers:
    :return:
    """

    common_eda_tasks(a_df, a_num_attr_list)
    print('\na_df[a_num_attr_list].describe():\n', a_df[a_num_attr_list].describe())
    check_out_skew_and_kurtosis(a_df[a_num_attr_list])
    check_out_multi_co_linearity_in_cap_x(a_df, a_num_attr_list)
    check_out_outliers_in_cap_x(a_df, a_num_attr_list, show_outliers=show_outliers)


def common_categorical_attr_eda_tasks(a_df, a_cat_attr_list):
    """

    :param a_df:
    :param a_cat_attr_list:
    :return:
    """

    common_eda_tasks(a_df, a_cat_attr_list)
    # explore_cardinality_of_categorical_attrs(a_df, a_cat_attr_list)


if __name__ == '__main__':
    pass
