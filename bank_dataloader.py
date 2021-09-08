#@title Load Bank dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    u'campaign', u'previous', u'emp.var.rate', u'cons.price.idx',
    u'cons.conf.idx', u'euribor3m', u'nr.employed', u'job_admin.',
    u'job_blue-collar', u'job_entrepreneur', u'job_housemaid',
    u'job_management', u'job_retired', u'job_self-employed', u'job_services',
    u'job_student', u'job_technician', u'job_unemployed', u'job_unknown',
    u'marital_divorced', u'marital_married', u'marital_single',
    u'marital_unknown', u'education_basic.4y', u'education_basic.6y',
    u'education_basic.9y', u'education_high.school', u'education_illiterate',
    u'education_professional.course', u'education_university.degree',
    u'education_unknown', u'default_no', u'default_unknown', u'default_yes',
    u'housing_no', u'housing_unknown', u'housing_yes', u'loan_no',
    u'loan_unknown', u'loan_yes', u'contact_cellular', u'contact_telephone',
    u'day_of_week_fri', u'day_of_week_mon', u'day_of_week_thu',
    u'day_of_week_tue', u'day_of_week_wed', u'poutcome_failure',
    u'poutcome_nonexistent', u'poutcome_success', u'y_yes', u'age_0', u'age_1',
    u'age_2', u'age_3', u'age_4', u'duration_0.0', u'duration_1.0',
    u'duration_2.0', u'duration_3.0', u'duration_4.0'
]
features = [
    u'campaign', u'previous', u'emp.var.rate', u'cons.price.idx',
    u'cons.conf.idx', u'euribor3m', u'nr.employed', u'job_admin.',
    u'job_blue-collar', u'job_entrepreneur', u'job_housemaid',
    u'job_management', u'job_retired', u'job_self-employed', u'job_services',
    u'job_student', u'job_technician', u'job_unemployed', u'job_unknown',
    u'marital_divorced', u'marital_married', u'marital_single',
    u'marital_unknown', u'education_basic.4y', u'education_basic.6y',
    u'education_basic.9y', u'education_high.school', u'education_illiterate',
    u'education_professional.course', u'education_university.degree',
    u'education_unknown', u'default_no', u'default_unknown', u'default_yes',
    u'housing_no', u'housing_unknown', u'housing_yes', u'loan_no',
    u'loan_unknown', u'loan_yes', u'contact_cellular', u'contact_telephone',
    u'day_of_week_fri', u'day_of_week_mon', u'day_of_week_thu',
    u'day_of_week_tue', u'day_of_week_wed', u'poutcome_failure',
    u'poutcome_nonexistent', u'poutcome_success', u'age_0', u'age_1',
    u'age_2', u'age_3', u'age_4', u'duration_0.0', u'duration_1.0',
    u'duration_2.0', u'duration_3.0', u'duration_4.0'
]
LABEL_COLUMN = ["y_yes"]
protected_features = ['marital_single', 'marital_married']


def get_bank_data():
  data_path = './data/bank-additional-full.csv'
  df = pd.read_csv(data_path, sep=';')
  continuous_features = [
      'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
      'euribor3m', 'nr.employed'
  ]
  continuous_to_categorical_features = ['age', 'duration']
  categorical_features = [
      'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
      'day_of_week', 'poutcome', 'y'
  ]

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_df, categorical_columns=[]):
    # Binarize categorical columns.
    binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
    return binarized_df

  def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
    input_df[continuous_column_name] = pd.cut(
        input_df[continuous_column_name], bins, labels=False)

  for c in continuous_to_categorical_features:
    b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 100]))
    bucketize_continuous_column(df, c, bins=b)

  df = binarize_categorical_columns(
      df,
      categorical_columns=categorical_features +
      continuous_to_categorical_features)

  to_fill = [
      u'duration_0.0', u'duration_1.0', u'duration_2.0', u'duration_3.0',
      u'duration_4.0'
  ]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

  normalize_features = [
      'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
  ]
  for feature in normalize_features:
    df[feature] = df[feature] - np.mean(df[feature])

  label = ["u'y_yes"]
  df = df[FEATURES]

  return df

def get_data():
    df = get_bank_data()

    y = np.array(df[LABEL_COLUMN]).flatten()

    X_train_bank, X_test_bank, y_train_bank, y_test_bank = train_test_split(df, y, test_size=0.2, random_state=42)
    protected_train_bank = [X_train_bank[g] for g in protected_features]
    protected_test_bank = [X_test_bank[g] for g in protected_features]
    X_train_bank = np.array(X_train_bank[features])
    X_test_bank = np.array(X_test_bank[features])

    return X_train_bank, y_train_bank, X_test_bank, y_test_bank, protected_train_bank, protected_test_bank