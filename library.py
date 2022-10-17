import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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
