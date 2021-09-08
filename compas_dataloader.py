import pandas as pd
import numpy as np


LABEL_COLUMN = 'two_year_recid'
PROTECTED_GROUPS = [
    'sex_Female', 'sex_Male'
]


def get_compas_data():
  data_path = './data/compas-scores-two-years.csv'
  df = pd.read_csv(data_path)
  FEATURES = [
      'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
      'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
      'two_year_recid'
  ]
  df = df[FEATURES]
  df = df[df.days_b_screening_arrest <= 30]
  df = df[df.days_b_screening_arrest >= -30]
  df = df[df.is_recid != -1]
  df = df[df.c_charge_degree != 'O']
  df = df[df.score_text != 'N/A']
  continuous_features = [
      'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
  ]
  continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
  categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_df, categorical_columns=[]):
    # Binarize categorical columns.
    binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
    return binarized_df

  def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
    input_df[continuous_column_name] = pd.cut(
        input_df[continuous_column_name], bins, labels=False)

  for c in continuous_to_categorical_features:
    b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
    if c == 'priors_count':
      b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
    bucketize_continuous_column(df, c, bins=b)

  df = binarize_categorical_columns(
      df,
      categorical_columns=categorical_features +
      continuous_to_categorical_features)

  to_fill = [
      u'decile_score_0', u'decile_score_1', u'decile_score_2',
      u'decile_score_3', u'decile_score_4', u'decile_score_5'
  ]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
  to_fill = [
      u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
      u'priors_count_3.0', u'priors_count_4.0'
  ]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

  features = [
      u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
      u'race_African-American', u'race_Asian', u'race_Caucasian',
      u'race_Hispanic', u'race_Native American', u'race_Other',
      u'score_text_High', u'score_text_Low', u'score_text_Medium',
      u'sex_Female', u'sex_Male', u'age_0', u'age_1', u'age_2', u'age_3',
      u'age_4', u'age_5', u'decile_score_0', u'decile_score_1',
      u'decile_score_2', u'decile_score_3', u'decile_score_4',
      u'decile_score_5', u'priors_count_0.0', u'priors_count_1.0',
      u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
  ]
  label = ['two_year_recid']

  df = df[features + label]
  return df, features, label


def get_data():
    df, feature_names, label_column = get_compas_data()

    from sklearn.utils import shuffle
    df = shuffle(df, random_state=12345)
    N = len(df)
    train_df = df[:int(N * 0.66)]
    test_df = df[int(N * 0.66):]

    X_train_compas = np.array(train_df[feature_names])
    y_train_compas = np.array(train_df[label_column]).flatten()
    X_test_compas = np.array(test_df[feature_names])
    y_test_compas = np.array(test_df[label_column]).flatten()

    protected_train_compas = [np.array(train_df[g]) for g in PROTECTED_GROUPS]
    protected_test_compas = [np.array(test_df[g]) for g in PROTECTED_GROUPS]

    return X_train_compas, y_train_compas, X_test_compas, y_test_compas, protected_train_compas, protected_test_compas
