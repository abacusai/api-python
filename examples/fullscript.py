from sys import stderr
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression


def transform_concrete(concrete_dataset):
  feature_df = concrete_dataset.drop(['flyash'], axis=1)
  no_flyash = feature_df[concrete_dataset.flyash == 0.0]
  flyash = feature_df[concrete_dataset.flyash > 0.0]
  mean_df = no_flyash.mean()
  print(mean_df)
  return pd.concat([no_flyash - no_flyash.assign(age=0).mean(), flyash - flyash.assign(age=0).mean()])


def to_quantiles(X):
  qt = QuantileTransformer(n_quantiles=20)
  X_q = qt.fit_transform(X.values)
  print(qt.quantiles_)
  return qt, X_q

def train_model(training_dataset):
  np.random.seed(5)

  X = training_dataset.drop(['csMPa'], axis=1)
  print(X.mean())
  y = training_dataset.csMPa
  qt, X_q = to_quantiles(X)

  recent_model = LinearRegression()
  fit_result = recent_model.fit(X_q, y)
  print(fit_result)
  model_r2 = recent_model.score(qt.transform(X.values), y)

  print(f'Linear model R^2 = {model_r2}')
  if model_r2 < 0.50:
    raise RuntimeError('Could not get a model with sufficient accuracy')
  if model_r2 < 0.85:
      print(f'Got a low R^2 {model_r2}', file=stderr)

  return (X.columns, qt, recent_model)


def predict(model, query):
  columns, qt, recent_model = model
  import pandas as pd
  X = pd.DataFrame({c: [query.get(c, 0.0)] for c in columns})
  y = recent_model.predict(qt.transform(X.values))[0]
  return {'csMPa': y}