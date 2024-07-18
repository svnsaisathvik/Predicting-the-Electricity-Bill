import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py

df=pd.read_csv('electricity_bill_dataset.csv')
df.head(5)
# sns.heatmap(data=df.corr(numeric_only=True),annot=True,cmap='viridis')
# print(df['MotorPump'].unique())
df.drop('MotorPump', axis=1, inplace=True)
# print(df['MotorPump'])
# print(df['City'].unique())
df=pd.get_dummies(df,columns=['City','Company'])
# print(df['Company'].unique())po[]
# print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('ElectricityBill',axis=1),df['ElectricityBill'], test_size=0.10)
# print(y_train)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
predictions = model.predict(X_test)
# print(predictions)
plt.scatter(y_test,predictions)
plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))