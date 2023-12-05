import numpy as np


def get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev):

    # binary uses the sklearn target encoder

    if fast_script_dev:
        common_hyp_param_grid = {
            'preprocessor__numerical__imputer__strategy': ['mean'],  # default
            'preprocessor__nominal__target_encoder__smooth': ['auto']  # default
        }
    else:
        common_hyp_param_grid = {
            'preprocessor__numerical__imputer__strategy': ['mean', 'median'],  # default
            'preprocessor__nominal__target_encoder__smooth': ['auto']  # default
        }

    return common_hyp_param_grid


def get_sgd_class_hyp_param_grid(alpha_points, l1_ratio_points, m_points, cap_x_df, binary, fast_script_dev):

    if binary:
        common_hyp_param_grid_binary = get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev)
    else:
        raise NotImplementedError

    early_stopping = [True]  # default=False
    n_iter_no_change = [5]  # default=5
    tol = [0.001]  # default=0.001
    validation_fraction = [0.1]  # default=0.1

    if fast_script_dev:
        sgd_class_hyp_param_grid = {
            # 'estimator__penalty': ['l2'],  # default
            # 'estimator__alpha': [0.0001],  # default
            # 'estimator__l1_ratio': [0.15],  # default
            # 'estimator__n_jobs': [None],  # default
            # early stopping params
            # 'estimator__early_stopping': early_stopping,
            # 'estimator__n_iter_no_change': n_iter_no_change,
            # 'estimator__tol': tol,  # default
            # 'estimator__validation_fraction': validation_fraction,
        }
    else:
        sgd_class_hyp_param_grid = {
            'estimator__penalty': ['l2'],  # default
            'estimator__alpha': [0.0001],  # default
            'estimator__l1_ratio': [0.15],  # default
            'estimator__n_jobs': [None],  # default
            # early stopping params
            # 'estimator__early_stopping': early_stopping,
            # 'estimator__n_iter_no_change': n_iter_no_change,
            # 'estimator__tol': tol,
            # 'estimator__validation_fraction': validation_fraction,
        }

    sgd_class_hyp_param_grid = sgd_class_hyp_param_grid | common_hyp_param_grid_binary

    return sgd_class_hyp_param_grid


def get_dt_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev):

    if binary:
        common_hyp_param_grid_binary = get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev)
    else:
        raise NotImplementedError

    if fast_script_dev:
        dt_class_hyp_param_grid = {
            # tree growth hyperparameters
            # 'estimator__criterion': ['gini'],  # default
            # 'estimator__splitter': ['best'],  # default
            # 'estimator__max_depth': [None],  # default
            # 'estimator__max_features': [None],  # default
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
        }
    else:
        dt_class_hyp_param_grid = {
            # tree growth hyperparameters
            'estimator__criterion': ['gini'],  # default
            'estimator__splitter': ['best'],  # default
            'estimator__max_depth': [None],  # default
            'estimator__max_features': [None],  # default
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
        }

    dt_class_hyp_param_grid = dt_class_hyp_param_grid | common_hyp_param_grid_binary

    return dt_class_hyp_param_grid


def get_rf_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev):

    if binary:
        common_hyp_param_grid_binary = get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev)
    else:
        raise NotImplementedError

    if fast_script_dev:
        rf_class_hyp_param_grid = {
            # tree growth hyperparameters
            # 'estimator__criterion': ['gini'],  # default
            # 'estimator__max_depth': [None],  # default
            # 'estimator__max_features': ['sqrt'],  # default
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
            # ensemble methods hyperparameters
            # 'estimator__n_estimators': [100],  # default
            # 'estimator__bootstrap': [True],  # default - do not work with bootstrap
            # 'estimator__oob_score': [False],  # default - do not work with oob_score
            # 'estimator__n_jobs': [-1],  # default=None
            # 'estimator__warm_start': [False],  # default - do not work with warm_start
            # 'estimator__max_samples': [None],  # default=None
            # other hyperparameters
            # 'estimator__verbose': [0],  # default - do not work with verbose
        }
    else:
        rf_class_hyp_param_grid = {
            # tree growth hyperparameters
            'estimator__criterion': ['gini', 'entropy'],  # default='gini'
            'estimator__max_depth': [None, 10, 20, 30],  # default=None
            'estimator__max_features': ['sqrt', 'log2', 0.5],  # default='sqrt'
            # 'estimator__min_samples_split': [2],  # do not work with min_samples_split
            # 'estimator__min_samples_leaf': [1],  # do not work with min_samples_leaf
            # 'estimator__min_weight_fraction_leaf': [0.0],  # do not work with min_weight_fraction_leaf
            # 'estimator__random_state': [None],  # set at instantiation in notebook
            # 'estimator__max_leaf_nodes': [None],  # do not work with max_leaf_nodes
            # 'estimator__min_impurity_decrease': [0.0],  # do not work with min_impurity_decrease
            # 'estimator__class_weight': [None],  # set at instantiation in notebook
            # 'estimator__ccp_alpha': [0.0]  # do not work with pruning
            # ensemble methods hyperparameters
            'estimator__n_estimators': [100, 200, 300],  # default=100
            # 'estimator__bootstrap': [True],  # default - do not work with bootstrap
            # 'estimator__oob_score': [False],  # default - do not work with oob_score
            'estimator__n_jobs': [None, -1],  # default=None
            # 'estimator__warm_start': [False],  # default - do not work with warm_start
            'estimator__max_samples': [None, 0.5, 0.7],  # default=None
            # other hyperparameters
            # 'estimator__verbose': [0],  # default - do not work with verbose
        }

    rf_class_hyp_param_grid = rf_class_hyp_param_grid | common_hyp_param_grid_binary

    return rf_class_hyp_param_grid


def get_ab_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev):

    if binary:
        common_hyp_param_grid_binary = get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev)
    else:
        raise NotImplementedError

    if fast_script_dev:
        ab_class_hyp_param_grid = {
        }
    else:
        ab_class_hyp_param_grid = {
        }

    ab_class_hyp_param_grid = ab_class_hyp_param_grid | common_hyp_param_grid_binary

    return ab_class_hyp_param_grid


def get_gb_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev):

    if binary:
        common_hyp_param_grid_binary = get_common_hyp_param_grid_binary(cap_x_df, m_points, fast_script_dev)
    else:
        raise NotImplementedError

    if fast_script_dev:
        gb_class_hyp_param_grid = {
        }
    else:
        gb_class_hyp_param_grid = {
        }

    gb_class_hyp_param_grid = gb_class_hyp_param_grid | common_hyp_param_grid_binary

    return gb_class_hyp_param_grid


def get_hyp_param_grid_func_dict(alpha_points, l1_ratio_points, m_points, cap_x_df, binary, fast_script_dev):

    hyp_param_grid_func_dict = {
        'SGDClassifier': get_sgd_class_hyp_param_grid(alpha_points, l1_ratio_points, m_points, cap_x_df, binary,
                                                      fast_script_dev),
        'DecisionTreeClassifier': get_dt_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev),
        'RandomForestClassifier': get_rf_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev),
        'AdaBoostClassifier': get_ab_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev),
        'GradientBoostingClassifier': get_gb_class_hyp_param_grid(m_points, cap_x_df, binary, fast_script_dev)
    }

    return hyp_param_grid_func_dict


def get_hyp_param_tuning_exp_dict(estimator_names, estimator_list, alpha_points, l1_ratio_points, m_points, cap_x_df,
                                  binary=True, fast_script_dev=False, print_param_grids=False):

    hyp_param_grid_func_dict = \
        get_hyp_param_grid_func_dict(alpha_points, l1_ratio_points, m_points, cap_x_df, binary, fast_script_dev)

    param_grid_list = []
    for estimator in estimator_names:
        param_grid_list.append(hyp_param_grid_func_dict[estimator])

    hyp_param_tuning_exp_dict = dict(zip(estimator_list, param_grid_list))

    if print_param_grids:
        for estimator_name, estimator_param_grid in hyp_param_tuning_exp_dict.items():
            print(f'\n', '*' * 60, sep='')
            print('*' * 60, sep='')
            print(f'{estimator_name}\n{estimator_param_grid}', sep='')

    return hyp_param_tuning_exp_dict


if __name__ == '__main__':
    pass
