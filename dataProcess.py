import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/dataSet.csv')

df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

df_train.to_csv("sData/df_train.csv", index=False)
df_valid.to_csv("sData/df_valid.csv", index=False)
df_test.to_csv("sData/df_test.csv", index=False)