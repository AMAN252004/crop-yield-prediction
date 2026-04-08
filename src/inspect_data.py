import pandas as pd

df = pd.read_csv("/Users/amanjakhar/Desktop/crop_yield_new/data/soybean_samples.csv")
print(df.columns)
print(df.head())
print(df.shape)