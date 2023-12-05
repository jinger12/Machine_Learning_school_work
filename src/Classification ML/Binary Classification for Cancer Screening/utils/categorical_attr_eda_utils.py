# global imports
import seaborn as sns
import matplotlib.pyplot as plt
# import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scikit_posthocs import posthoc_dunn

# local imports
import utils.attr_eda_utils as eda_utils
from utils.all_attr_eda_utils import check_out_target_imbalance, check_out_target_distribution
from utils.all_attr_eda_utils import convert_symmetric_matrix_to_df


def print_catplots(a_df, a_cat_attr_list, a_target_attr, a_kinds_list, num_unique_levels_threshold=18,
                   num_obs_threshold=1000):
    """

    :param num_obs_threshold:
    :param num_unique_levels_threshold:
    :param a_kinds_list:
    :param a_df:
    :param a_cat_attr_list:
    :param a_target_attr:
    :return:
    """

    if a_df.shape[0] > num_obs_threshold:
        print('\n', f'too many observations for other kinds of plots - only plot strip plots', sep='')
        a_kinds_list = ['box', 'strip']

    for attr in a_cat_attr_list:
        print('\n\n', 50 * '*', '\n', 50 * '*', sep='')
        print(attr)
        num_unique_levels = a_df[attr].nunique()
        print('\na_df[attr].nunique():', num_unique_levels, sep='')
        print('\na_df[attr].value_counts(dropna=False):\n', a_df[attr].value_counts(dropna=False), sep='')
        if num_unique_levels > num_unique_levels_threshold:
            print('\n', f'num_unique_levels = {num_unique_levels} which exceeds the num_unique_levels_threshold '
                        f'{num_unique_levels_threshold} - do not plot!', sep='')
            do_kruskal_wallis(a_df, attr, a_target_attr)
            continue
        for kind in a_kinds_list:
            print('\nkind of catplot:', kind)
            print(f'\ndrop rows with nan in {attr} attribute - assign to no_nan_df')
            no_nan_df = a_df.dropna()
            if kind == 'violin':
                sns.catplot(x=attr, y=a_target_attr, kind=kind, inner='stick', data=no_nan_df)
            else:
                sns.catplot(x=attr, y=a_target_attr, kind=kind, data=no_nan_df)
            plt.xticks(rotation=90)
            plt.grid()
            plt.show()

        # anova - assumptions for this test to be valid are not met by most observational data
        # do_one_way_anova(a_df, attr, a_target_attr)

        # kruskal-wallis
        do_kruskal_wallis(a_df, attr, a_target_attr)


def drop_categories_with_lt_n_instances(a_df, attr, a_target_attr, n):

    print(f'\ncheck category counts and drop categories with count < {n}')
    cat_drop_list = []
    for category in a_df[attr].unique():
        value_count = a_df.loc[a_df[attr] == category, a_target_attr].shape[0]
        if value_count < n:
            print(f'   category {category} has value count = {value_count} - drop it')
            cat_drop_list.append(category)

    a_df = a_df[~a_df[attr].isin(cat_drop_list)]

    return a_df


def do_kruskal_wallis(a_df, attr, a_target_attr):

    print(f'\nperform the kruskal-wallis test to understand if there is a difference in {a_target_attr} means between '
          f'the categories:')

    a_df = a_df.loc[:, [attr, a_target_attr]]
    a_df = a_df.dropna()
    a_df = drop_categories_with_lt_n_instances(a_df, attr, a_target_attr, 5)

    groups = [a_df.loc[a_df[attr] == group, a_target_attr].values for group in a_df[attr].unique()]
    if len(groups) < 2:
        print(f'\nnumber of groups is {len(groups)} - need at least two groups in stats.kruskal()')
        return None
    results = stats.kruskal(*groups)

    kruskal_wallis_alpha = 0.05
    dunns_test_alpha = 0.05
    print(f'\n   kruskal-wallis p-value: {results.pvalue}', sep='')
    if results.pvalue < kruskal_wallis_alpha:
        print(f'   at least one mean is different then the others at alpha = {kruskal_wallis_alpha} level - conduct '
              f'the dunn\'s test')
        results = posthoc_dunn(a_df, val_col=a_target_attr, group_col=attr, p_adjust='bonferroni')
        sym_matrix_df = convert_symmetric_matrix_to_df(results, 'p_value')
        sym_matrix_df = sym_matrix_df[sym_matrix_df.p_value < dunns_test_alpha]
        print(f'\ndunn\'s test results:')
        print(sym_matrix_df)
    else:
        print(f'\n   differences in means are not significant at alpha = {kruskal_wallis_alpha} level')


def do_one_way_anova(a_df, attr, a_target_attr):

    temp_df = pd.DataFrame(
        {'y': a_df[a_target_attr].values,
         'x': a_df[attr]}
    )

    print(f'\nSimple Linear Regression of {a_target_attr} on {attr}:\n', sep='')
    mod = ols('y ~ x', data=temp_df).fit()
    print(mod.summary())

    print(f'\nOne way ANOVA of {a_target_attr} / {attr}:\n', sep='')
    print(sm.stats.anova_lm(mod, typ=1))


def categorical_attr_eda_regression(a_df, a_cat_attr_list, a_target_attr):
    """

    :param a_cat_attr_list:
    :param a_target_attr:
    :param a_df:
    :return:
    """

    print('\n', 60 * '*', sep='')
    print('categorical_attr_eda_regression - bivariate analysis: target attribute vs categorical attributes')

    kinds_list = ['swarm', 'box', 'violin']
    print_catplots(a_df, a_cat_attr_list, a_target_attr, kinds_list)


def categorical_attr_eda_classification(a_df, a_cat_attr_list, a_target_attr, plt_threshold=10):
    """

    :param plt_threshold:
    :param a_cat_attr_list:
    :param a_target_attr:
    :param a_df:
    :return:
    """
    # https://seaborn.pydata.org/tutorial/categorical.html
    print('\n', 60 * '*', sep='')
    print('categorical_attr_eda_classification - bivariate analysis: target attribute vs categorical attributes')

    for attr in a_cat_attr_list:
        print('\n', 40 * '*', sep='')
        print(f'attr: {attr}')
        n_unique_levels = a_df[attr].nunique()
        print(f'\nnumber of levels in the categorical attribute: {n_unique_levels}')
        print(f'\nlevels in the categorical attribute: {a_df[attr].unique()}')
        print(f'\nlevels balance in the categorical attribute:\n{a_df[attr].value_counts(normalize=True)}')
        if n_unique_levels < plt_threshold:
            sns.catplot(x=attr, hue=a_target_attr, kind='count', data=a_df)
            plt.grid()
            plt.show()
            sns.catplot(x=a_target_attr, hue=attr, kind='count', data=a_df)
            plt.grid()
            plt.show()
        else:
            print(f'\nNumber of levels in the categorical attribute is {n_unique_levels} which is greater than the '
                  f'plt_threshold = {plt_threshold} - do not plot')


def categorical_attr_eda(a_df, a_cat_attr_list, a_prediction_task_type, a_target_attr):
    """

    :param a_target_attr:
    :param a_df:
    :param a_cat_attr_list:
    :param a_prediction_task_type:
    :return:
    """
    print('\n', 80 * '*', sep='')
    print('categorical_attr_eda')
    print('a_prediction_task_type:', a_prediction_task_type, '\n')

    if len(a_cat_attr_list) == 0:
        print(f'\nthere are no categorical attributes')
        return None

    eda_utils.common_categorical_attr_eda_tasks(a_df, a_cat_attr_list)

    if a_prediction_task_type == 'regression':
        check_out_target_distribution(a_df, a_target_attr)
        categorical_attr_eda_regression(a_df, a_cat_attr_list, a_target_attr)
    elif a_prediction_task_type == 'classification':
        check_out_target_imbalance(a_df, a_target_attr)
        categorical_attr_eda_classification(a_df, a_cat_attr_list, a_target_attr)
    else:
        raise TypeError('unrecognized prediction_type_task!!!')


if __name__ == '__main__':
    pass
