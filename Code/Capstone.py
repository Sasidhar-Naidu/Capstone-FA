import os
import pandas as pd

df = pd.read_csv(r"Code\Dataset\financialanalytics_Dataset.csv")

print(df.head())
print(df.describe())
