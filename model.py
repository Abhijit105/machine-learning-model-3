import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score

#########################################################################################

df = pd.read_csv("./house_prices_ml_ready.csv")
print("\n##### Data 5 sample records: #####")
print(df.sample(5))

print("\n##### Data shape: #####")
print(df.shape)

print("\n##### Data size: #####")
print(df.size)

print("\n##### Data column names: #####")
print(df.columns)

#########################################################################################

encoder = OneHotEncoder()
encoded_locations = encoder.fit_transform(df[['location']])
encoded_feature_names = encoder.get_feature_names_out(['location'])
df_encoded_locations = pd.DataFrame(encoded_locations.toarray(), columns=encoded_feature_names)
df_original = df.copy()
df = df.drop(columns=['location'], axis=1)
df = pd.concat([df, df_encoded_locations], axis=1)

#########################################################################################

X = df.drop(['house_price'], axis=1)
y = df['house_price']
print("\n##### X shape: #####")
print(X.shape)

print("\n##### y shape: #####")
print(y.shape)

#########################################################################################

model_linear_regression = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model_linear_regression.fit(X_train, y_train)
y_pred = model_linear_regression.predict(X_test)

rmse_linear_regression = root_mean_squared_error(y_test, y_pred)
r2_linear_regression = r2_score(y_test, y_pred)

print("\n##### RMSE: #####")
print(rmse_linear_regression)

print("\n##### R2 score: #####")
print(r2_linear_regression)

#########################################################################################