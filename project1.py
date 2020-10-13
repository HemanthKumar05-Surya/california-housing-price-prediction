                     # California Housing Price Prediction

#Required libraries for this problem
import pandas as pd
import matplotlib.pyplot as plt

#1) Loading the data,from an excel file into this program
dataset=pd.read_excel('housing.xlsx') #dataset contains 20640*10 Features
#printing the first few rows of this data eg., taking first 20 rows
df=dataset.head(n=20)
#there are 20640 instances & "total_bedrooms" has 20433 non_null values
#207 values are missing
dataset.info() #Details of the dataset
dataset.isnull().sum()#checking null values
dataset['ocean_proximity'].value_counts() #Getting total instances for this column
dataset.describe() #checking mean,median,min,max,quartile and so on for dataset

#Extracting input(X) and output(Y) from dataset

X=dataset.iloc[:,0:9].values
Y=dataset.iloc[:,9].values

# 2) Handling missing values
# using Imputer form sckitlearn library
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[4]])
X[:,[4]]=imputer.fit_transform(X[:,[4]])
df1=pd.DataFrame(X) # null values replaced with mean
df1.isnull().sum()

#3)Encoding categorical data
# using labelencoder and onehotencoder from sckitlearn library
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,8]=labelencoder_X.fit_transform(X[:,8])# converting into numeric data
df1=pd.DataFrame(X)
df1.isnull().sum()

onehotencoder=OneHotEncoder(categorical_features=[8])
X=onehotencoder.fit_transform(X).toarray()

#4) Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#5) Standardizing the data


regressor.score(X_train,Y_train)
regressor.score(X_test,Y_test)
predicted=regressor.predict(X_test)
expected=Y_test

#RMSE from LinearRegression
import numpy as np
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(expected,predicted))
plt.scatter(expected,predicted)
plt.plot([0,1000],[0,1000],'--k')
plt.xlabel('Expected price')
plt.ylabel('Predicted price')
plt.title('California Housing Price Prediction')
plt.show()


