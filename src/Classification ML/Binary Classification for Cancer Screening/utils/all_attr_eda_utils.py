# global imports
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np

# local imports
import utils.lin_reg_diag_utils as lin_reg_diag_utils
import utils.attr_eda_utils as attr_eda_utils


def convert_symmetric_matrix_to_df(sym_matrix, label):

    sym_matrix_df = (
        sym_matrix.where(
            np.triu(np.ones(sym_matrix.shape), k=1)
            .astype(bool)
        )
        .stack()
        .to_frame(name=label)
    )

    new_index = [str(i) + '_' + str(j) for i, j in sym_matrix_df.index]
    sym_matrix_df.index = new_index
    sym_matrix_df = sym_matrix_df.sort_values(label, ascending=False)

    return sym_matrix_df


def check_out_non_ml_attrs(a_df, a_non_ml_attr_list):
    """

    :param a_df:
    :param a_non_ml_attr_list:
    :return:
    """

    print('\n', 60 * '*', sep='')
    print('check_out_non_ml_attrs:')

    for attr in a_non_ml_attr_list:
        print('\n', 20 * '*', sep='')
        print('attr:', attr)
        print('attr dtype:', a_df[attr].dtype)
        print('a_df.shape:', a_df.shape)
        print('a_df[attr].nunique():', a_df[attr].nunique())
        print('a_df[attr].head():\n', a_df[attr].head(), sep='')


def check_for_duplicate_observations(a_df):
    print('\n', 60 * '*', sep='')
    print('check_for_duplicate_observations:')

    dedup_a_df = a_df.drop_duplicates()
    print('a_df.shape:', a_df.shape)
    print('dedup_a_df.shape:', dedup_a_df.shape)

    if dedup_a_df.shape[0] < a_df.shape[0]:
        print('caution: data set contains duplicate observations!!!')
    else:
        print('no duplicate observations observed in data set')


def check_out_missingness(a_df, sample_size_threshold=250, verbose=True, nullity_corr_method='spearman',
                          nullity_corr_threshold=0.75):
    """

    :param nullity_corr_threshold:
    :param nullity_corr_method:
    :param sample_size_threshold:
    :param verbose:
    :param a_df:
    :return:
    """
    print('\n', 60 * '*', sep='')
    print('check_out_missingness:')

    if verbose:
        print('\nNA (np.nan or None) count - a_df[an_attr_list].isna().sum():\n', a_df.isna().sum(), sep='')
        print('\nNA (np.nan or None) fraction - a_df[an_attr_list].isna().sum() / a_df.shape[0]:\n',
              a_df.isna().sum() / a_df.shape[0], sep='')

    if a_df.isna().sum().sum() > 0:
        print('\nmissing values in data set!!!')
        sample_size = a_df.shape[0]
        if sample_size > sample_size_threshold:
            sample_size = 250
        print('\nuse missingno to understand pattern of missingness')
        print('a_df.shape[0]:', a_df.shape[0])
        print('missingno sample_size:', sample_size)
        msno.matrix(a_df.sample(sample_size, random_state=42))
        plt.show()
        msno.heatmap(a_df.sample(sample_size, random_state=42))
        plt.show()
        print('\n', f'nullity correlation using the {nullity_corr_method} method with corr row threshold '
                    f'{nullity_corr_threshold}', sep='')
        _ = attr_eda_utils.get_flattened_corr_matrix(a_df.isnull().corr(method=nullity_corr_method),
                                                     corr_threshold=nullity_corr_threshold)

    else:
        print('\nno missing values in data set.')


def all_attr_eda(a_df, a_non_ml_attr_list, a_prediction_task_type):
    """

    :param a_df:
    :param a_non_ml_attr_list:
    :param a_prediction_task_type:
    :return:
    """

    print('\n', 80 * '*', sep='')
    print('all_attr_eda')
    print('a_prediction_task_type:', a_prediction_task_type, '\n')

    # check out non_ml_attr
    check_out_non_ml_attrs(a_df, a_non_ml_attr_list)

    # check for duplicates
    check_for_duplicate_observations(a_df)

    # check out missingness
    check_out_missingness(a_df)


def check_out_target_imbalance(a_df, a_target_attr):
    """

    :param a_df:
    :param a_target_attr:
    :return:
    """
    print('\n', 60 * '*', sep='')
    print('check_out_target_distribution:')

    print(f'\nnumber of classes in target attribute: {a_df[a_target_attr].nunique()}')
    print(f'\nclasses in target attribute: {a_df[a_target_attr].unique()}')
    print(f'\nclass balance:\n{a_df[a_target_attr].value_counts(normalize=True)}')


def check_out_target_distribution(a_df, a_target_attr):
    """

    :param a_df:
    :param a_target_attr:
    :return:
    """

    print('\n', 60 * '*', sep='')
    print('check_out_target_distribution:')
    print('\na_df[a_target_attr].describe():\n', a_df[a_target_attr].describe(), sep='')

    nan_count = a_df[a_target_attr].isna().sum()
    if nan_count > 0:
        print('\n\n', '*' * 20, '\n', '*' * 20, '\n', '*' * 20, sep='')
        print(f'there are {nan_count} nans in the target vector!!!')
        print('*' * 20, '\n', '*' * 20, '\n', '*' * 20, sep='')
    print('\n')
    a_df[a_target_attr].hist()
    plt.grid()
    plt.show()
    lin_reg_diag_utils.test_for_normality(a_df[a_target_attr])


def check_for_complete_unique_attrs(cap_x_df):

    concern_list = []
    for attr in cap_x_df.columns:
        label = ''
        if cap_x_df[attr].nunique() == cap_x_df.shape[0]:
            label = 'examine more closely'
            concern_list.append(attr)
        print(f'{attr}; {cap_x_df[attr].nunique()}; {cap_x_df[attr].dtype}; {cap_x_df.shape[0]}',
              f'{label}')

    return concern_list


if __name__ == '__main__':
    pass
