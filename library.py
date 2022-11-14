import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class MappingTransformer(BaseEstimator, TransformerMixin):  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_
  
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class RenamingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    keys = set(self.mapping_dict.keys())
    cols = set(X.columns.to_list())
    diff = keys - cols
    print(diff)

    assert len(diff) == 0 , f'{self.__class__.__name__}.transform unknown column "{diff}"'  #column legit?
    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    target_column = set([self.target_column])
    cols = set(X.columns.to_list())
    diff = target_column - cols

    assert len(diff) == 0 , f'{self.__class__.__name__}.transform unknown column "{diff}"'
    X_ = X.copy()
    X_ = pd.get_dummies(X_,
                  prefix=self.target_column,    #your choice
                  prefix_sep='_',     #your choice
                  columns=[self.target_column],
                  dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                  drop_first=self.drop_first    #really should be True but could screw us up later
                  )
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'{self.__class__.__name__} constructor expected list but got {type(column_list)} instead.'
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    column_list = set(self.column_list)
    cols = set(X.columns.to_list())
    diff = column_list - cols
    if len(diff) != 0:
      assert self.action != 'keep', f'{self.__class__.__name__} does not contain these columns to keep: {diff}'
      if self.action == 'drop':
        print(f"\nWarning: {self.__class__.__name__} does not contain these columns to drop: {diff}.\n")
    X_ = X.copy()
    if self.action == 'keep':
      diff = cols - column_list 
      X_.drop(diff, errors='ignore', axis=1, inplace=True)
    else:
      X_.drop(self.column_list, errors='ignore', axis=1, inplace=True)
    return X_
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class PearsonTransformer(BaseEstimator, TransformerMixin): 
  def __init__(self, threshold):
    assert isinstance(threshold, float), f'{self.__class__.__name__} constructor expected float but got {type(threshold)} instead.'
    self.threshold = threshold

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    df_corr = X.corr(method='pearson')

    abs_df = df_corr.mask(df_corr[df_corr.columns].abs() > self.threshold, True)
    masked_df = abs_df.mask(abs_df <= self.threshold, False)

    upper_mask = np.triu(masked_df, k=1)
    upper_mask[np.where(upper_mask==0)]=False

    correlated_columns = [i[1] for i in enumerate(masked_df.columns) if True in upper_mask[:, i[0]]]

    new_df = transformed_df.drop(columns=correlated_columns)

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name):
    assert isinstance(column_name, str), f'{self.__class__.__name__} constructor expected string but got {type(column_name)} instead.'
    self.column_name = column_name

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.column_name in X.columns.to_list(), f'unknown column {self.column_name}'
    assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

    #your code below
    m = X[self.column_name].mean()
    sig = X[self.column_name].std()
    s3max = m + 3 * sig
    s3min = m - 3 * sig

    transformed_df_ = transformed_df.copy()
    transformed_df_[self.column_name] = X[self.column_name].clip(lower=s3min, upper=s3max)

    return transformed_df_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence):
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'
    assert isinstance(fence, str), f'{self.__class__.__name__} constructor expected string but got {type(fence)} instead.'
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])

    X_ = X.copy()

    q1 = X[self.target_column].quantile(0.25)
    q3 = X[self.target_column].quantile(0.75)
    iqr = q3-q1
    if(self.fence == 'inner'):
      inner_low = q1-1.5*iqr
      inner_high = q3+1.5*iqr
      X_[self.target_column] = X[self.target_column].clip(lower=inner_low, upper=inner_high)
    elif(self.fence == 'outer'):
      outer_low = q1-3*iqr
      outer_high = q3+3*iqr
      X_[self.target_column] = X[self.target_column].clip(lower=outer_low, upper=outer_high)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    scaler =  MinMaxScaler()
    columns_X = X.columns.to_list()
    numpy_result = scaler.fit_transform(X)  #does not return a dataframe!
    new_df = pd.DataFrame(numpy_result, columns=columns_X)  #turn it back into a dataframe
    new_df.describe(include='all').T

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform"):
    assert isinstance(weights, str), f'{self.__class__.__name__} constructor expected string but got {type(weights)} instead.'
    assert isinstance(n_neighbors, int), f'{self.__class__.__name__} constructor expected int but got {type(n_neighbors)} instead.'
    self.n_neighbors = n_neighbors
    self.weights = weights

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    columns = list(X.columns)

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=self.n_neighbors,        #a rough guess
                        weights=self.weights,                 #could alternatively have distance factor in
                        add_indicator=False)                  #do not add extra column for NaN

    imputed_data = imputer.fit_transform(X)  #instantiate

    result_df = pd.DataFrame(imputed_data, columns = columns)
    return result_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5)
  var = [] 
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  
      train_pred = model.predict(train_X)     
      test_pred = model.predict(test_X)      
      train_f1 = f1_score(train_y, train_pred)
      test_f1 = f1_score(test_y, test_pred)     
      f1_ratio = test_f1/train_f1          
      var.append(f1_ratio)

  rs_value = sum(var)/len(var) 
  idx = np.array(abs(var - rs_value)).argmin() 
  return idx


def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below

  features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()
  
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)

  X_test_transformed = the_transformer.fit_transform(X_test)

  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy 


titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                           transformer,
                                                                           rs, ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy


customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)

def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                           transformer,
                                                                           rs, ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy


def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}

  result_df = result_df.round(2)

  #Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
  #Note that fancy_df is not really a dataframe. More like a printable object.
  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)

def halving_search(model, grid, x_train, y_train, factor=3, scoring='roc_auc'):
  #your code below
  halving_cv = HalvingGridSearchCV(
    model, grid,  #our model and the parameter combos we want to try
    scoring=scoring,  #could alternatively choose f1, accuracy or others
    n_jobs=-1,
    min_resources="exhaust",
    factor=factor,  #a typical place to start so triple samples and take top 3rd of combos on each iteration
    cv=5, random_state=1234,
    refit=True,  #remembers the best combo and gives us back that model already trained and ready for testing
  )

  grid_result = halving_cv.fit(x_train, y_train)
  return grid_result
