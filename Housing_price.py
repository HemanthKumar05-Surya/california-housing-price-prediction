                   #California Housing Price Prediction
# Requited libraries for this problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

1) Loading the data
dataset=pd.read_excel('housing.xlsx')
#Printing first few rows of the data
df=dataset.head(n=20)
X=dataset.iloc[:,0:9].values
Y=dataset.iloc[:,9].values

dataset.isnull().sum()

2)Handling the missing values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[4]])
X[:,[4]]=imputer.fit_transform(X[:,[4]])
df1=pd.DataFrame(X).values
df1.isnull().sum()

3) Encoding the categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_df1=LabelEncoder()
df1[:,8]=labelencoder_df1.fit_transform(df1[:,8])
df2=pd.DataFrame(df1)
df2.isnull().sum()

onehotencoder=OneHotEncoder(categorical_features=[8])
df2=onehotencoder.fit_transform(df2).toarray()

4) Splitting the dataset

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

5) Standardizing the data

from sklearn.preprocessing import StandardScaler
sc_df2=StandardScaler()
X_train=sc_df2.fit_transform(X_train)
X_test=sc_df2.fit_transform(X_test)

6) Performing the linear Regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
regressor.coef_
regressor.intercept_
regressor.score(X_train,Y_train)

predicted=regressor.predict(X_test)
expected=Y_test

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(expected,predicted))
plt.scatter(expected,predicted,color='blue')
plt.plot([0,600000],[-0,600000],'--k',color='black')
plt.title('California Housing Price Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Expected Price')
plt.show()




