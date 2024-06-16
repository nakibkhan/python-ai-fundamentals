import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False, as_frame=False)

df = pd.DataFrame(data=california_housing.data, columns= california_housing.feature_names)

df['target'] = california_housing.target

# Display basic information about the dataset
print(df.head())
# print(df.describe())