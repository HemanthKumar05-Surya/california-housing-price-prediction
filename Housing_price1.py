          # California Housing Price Prediction # Bonus exercise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_excel('housing1.xlsx')
X=dataset.iloc[:,[0]]
Y=dataset.iloc[:,[1]]

dataset.isnull().sum()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
regressor.coef_
regressor.intercept_
y_pred=regressor.predict(X_test)

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
regressor.score(X_train,Y_train)
plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_test,y_pred,color='black')
plt.xlabel('median house value')
plt.ylabel('median income')
plt.title('California Housing Price Prediction')
plt.show()


