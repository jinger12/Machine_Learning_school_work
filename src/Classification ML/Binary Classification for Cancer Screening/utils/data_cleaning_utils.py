import pandas as pd
# import sys


def select_numerical_data(cap_x_df, y_df):

    df = pd.concat([cap_x_df, y_df], axis=1)

    print('\n', df.shape)
    print('\n', df.dtypes)

    df = df.select_dtypes(include=['float', 'int'])

    print('\n', df.shape)
    print('\n', df.dtypes)

    cap_x_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]

    return cap_x_df, y_df


def drop_obs_with_nan_response(cap_x_df, y_df):

    df = pd.concat([cap_x_df, y_df], axis=1)

    print('\n', df.shape)

    response_name = y_df.name

    df = df.dropna(subset=[response_name])

    print('\n', df.shape)

    cap_x_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]

    return cap_x_df, y_df


def drop_useless_attr(cap_x_df, y_df, useless_attr):

    df = pd.concat([cap_x_df, y_df], axis=1)

    df = df.drop(labels=useless_attr, axis=1)

    cap_x_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]

    return cap_x_df, y_df


def drop_cat_disguised_as_num_attr(cap_x_df, y_df, cat_disguised_as_num):

    df = pd.concat([cap_x_df, y_df], axis=1)

    df = df.drop(labels=cat_disguised_as_num, axis=1)

    cap_x_df = df.iloc[:, 0:-1]
    y_df = df.iloc[:, -1]

    return cap_x_df, y_df


def get_missingness_drop_list(cap_x_df, missingness_threshold=0.20):

    drop_list = []
    for attr in cap_x_df.iloc[:, :-1].columns:
        attr_missingness = cap_x_df[attr].isna().sum()/cap_x_df.shape[0]
        if attr_missingness > missingness_threshold:
            drop_list.append(attr)

    return drop_list


def drop_attrs_gt_missingness_threshold(cap_x_df, missingness_threshold=0.20):

    drop_list = get_missingness_drop_list(cap_x_df, missingness_threshold)
    cap_x_df = cap_x_df.drop(columns=drop_list)

    return cap_x_df, drop_list


if __name__ == "__main__":
    pass
