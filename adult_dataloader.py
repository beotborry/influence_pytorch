import numpy as np
import pandas as pd
from torch.utils.data import Dataset

CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country'
]
CONTINUOUS_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
]
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
LABEL_COLUMN = 'label'

PROTECTED_GROUPS = [
    'gender_Female', 'gender_Male'
]


def get_adult_data():
    train_file = "./data/adult_train.csv"
    test_file = "./data/adult_test.csv"

    train_df_raw = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    test_df_raw = pd.read_csv(
        test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

    train_df_raw[LABEL_COLUMN] = (
        train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    test_df_raw[LABEL_COLUMN] = (
        test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    # Preprocessing Features
    pd.options.mode.chained_assignment = None  # default='warn'

    # Functions for preprocessing categorical and continuous columns.
    def binarize_categorical_columns(input_train_df,
                                     input_test_df,
                                     categorical_columns=[]):

        def fix_columns(input_train_df, input_test_df):
            test_df_missing_cols = set(input_train_df.columns) - set(
                input_test_df.columns)
            for c in test_df_missing_cols:
                input_test_df[c] = 0
            train_df_missing_cols = set(input_test_df.columns) - set(
                input_train_df.columns)
            for c in train_df_missing_cols:
                input_train_df[c] = 0
            input_train_df = input_train_df[input_test_df.columns]
            return input_train_df, input_test_df

        # Binarize categorical columns.
        binarized_train_df = pd.get_dummies(
            input_train_df, columns=categorical_columns)
        binarized_test_df = pd.get_dummies(
            input_test_df, columns=categorical_columns)
        # Make sure the train and test dataframes have the same binarized columns.
        fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                    binarized_test_df)
        return fixed_train_df, fixed_test_df

    def bucketize_continuous_column(input_train_df,
                                    input_test_df,
                                    continuous_column_name,
                                    num_quantiles=None,
                                    bins=None):
        assert (num_quantiles is None or bins is None)
        if num_quantiles is not None:
            train_quantized, bins_quantized = pd.qcut(
                input_train_df[continuous_column_name],
                num_quantiles,
                retbins=True,
                labels=False)
            input_train_df[continuous_column_name] = pd.cut(
                input_train_df[continuous_column_name], bins_quantized, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
                input_test_df[continuous_column_name], bins_quantized, labels=False)
        elif bins is not None:
            input_train_df[continuous_column_name] = pd.cut(
                input_train_df[continuous_column_name], bins, labels=False)
            input_test_df[continuous_column_name] = pd.cut(
                input_test_df[continuous_column_name], bins, labels=False)

    # Filter out all columns except the ones specified.
    train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                            [LABEL_COLUMN]]
    test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                          [LABEL_COLUMN]]
    # Bucketize continuous columns.
    bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
    bucketize_continuous_column(
        train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
    bucketize_continuous_column(
        train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
    bucketize_continuous_column(
        train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
    bucketize_continuous_column(
        train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
    train_df, test_df = binarize_categorical_columns(
        train_df,
        test_df,
        categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
    feature_names = list(train_df.keys())
    feature_names.remove(LABEL_COLUMN)
    num_features = len(feature_names)
    return train_df, test_df, feature_names


def get_data():
    train_df, test_df, feature_names = get_adult_data()
    X_train_adult = np.array(train_df[feature_names])
    y_train_adult = np.array(train_df[LABEL_COLUMN])
    X_test_adult = np.array(test_df[feature_names])
    y_test_adult = np.array(test_df[LABEL_COLUMN])

    #print(list(train_df[feature_names].columns).index("gender_Female")) # 58th column => female
    #print(list(train_df[feature_names].columns).index("gender_Male")) # 59th column => female

    protected_train_adult = [np.array(train_df[g]) for g in PROTECTED_GROUPS]
    protected_test_adult = [np.array(test_df[g]) for g in PROTECTED_GROUPS]
    return X_train_adult, y_train_adult, X_test_adult, y_test_adult, protected_train_adult, protected_test_adult

class CustomDataset(Dataset):
    def __init__(self, X_train_tensor, y_train_tensor):
        self.X_train = X_train_tensor
        self.y_train = y_train_tensor

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]