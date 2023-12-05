from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper
import pandas as pd


def inspect_trained_preprocessor(trained_preprocessor):
    for key, value in trained_preprocessor.__dict__.items():
        print('**********')
        print(key, value)


def get_input_num_and_nom_attr_names(trained_preprocessor):
    attr_names_list = trained_preprocessor.__dict__['_columns']
    num_attr_names = attr_names_list[0]
    nom_attr_names = attr_names_list[1]
    return num_attr_names, nom_attr_names


def get_multi_class_class_list(y_df, target_attr):
    class_list = y_df[target_attr].unique()
    class_list = class_list[1:]
    return class_list


def cat_enc_mul_cla_tar_enc_in_pipeline_col_name_util(trained_preprocessor, y_df, target_attr, inspect_preproc=False):

    # category_encoder multi class target encoder in sklearn pipeline column naming util

    if inspect_preproc:
        inspect_trained_preprocessor(trained_preprocessor)

    num_attr_names, nom_attr_names = get_input_num_and_nom_attr_names(trained_preprocessor)
    class_list = get_multi_class_class_list(y_df, target_attr)

    te_nom_attr_name_list = []
    for class_ in class_list:
        for attr in nom_attr_names:
            te_nom_attr_name_list.append(attr + '_' + str(class_))

    col_name_list = num_attr_names + te_nom_attr_name_list

    return col_name_list


def get_cat_enc_mul_cla_tar_enc_preproc_pipeline(numerical_attr, nominal_attr, num_trans_name='numerical',
                                                 nom_trans_name='nominal'):
    # build category_encoder multi class target encoder in sklearn preproc pipeline

    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()),
               ("scaler", StandardScaler())]
    )

    nominal_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               (
                   'target_encoder',
                   PolynomialWrapper(TargetEncoder())
               ),
               ("scaler", StandardScaler())
               ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (num_trans_name, numerical_transformer, numerical_attr),
            (nom_trans_name, nominal_transformer, nominal_attr)
        ]
    )

    return {
        'preprocessor': preprocessor,
        'num_trans_name': num_trans_name,
        'nom_trans_name': nom_trans_name
    }


def check_work(numerical_attr, nominal_attr, cap_x_df, y_df, target_attr):

    # this function compares the te preproc output from pipeline to the output without pipeline - are we doing things
    # right?

    # no pipeline
    no_pipeline_poly_te_np_df = PolynomialWrapper(TargetEncoder(cols=nominal_attr)).\
        fit_transform(cap_x_df, y_df[target_attr])
    no_pipeline_poly_te_np = StandardScaler().fit_transform(no_pipeline_poly_te_np_df)
    no_pipeline_poly_te_np_df = pd.DataFrame(
        data=no_pipeline_poly_te_np,
        columns=no_pipeline_poly_te_np_df.columns,
        index=no_pipeline_poly_te_np_df.index
    )

    # with pipeline
    preproc_dict = get_cat_enc_mul_cla_tar_enc_preproc_pipeline(numerical_attr, nominal_attr)
    preprocessor = preproc_dict['preprocessor']
    pipeline_poly_te_np = preprocessor.fit_transform(cap_x_df, y_df[target_attr])
    col_name_list = cat_enc_mul_cla_tar_enc_in_pipeline_col_name_util(preprocessor, y_df, target_attr)
    pipeline_poly_te_np_df = pd.DataFrame(
        data=pipeline_poly_te_np,
        columns=col_name_list,
        index=cap_x_df.index
    )

    # compare them
    print(f'We doing it right? {pipeline_poly_te_np_df.equals(no_pipeline_poly_te_np_df)}')


if __name__ == "__main__":
    pass
