import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30)

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test) 

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test,y_pred)

plt.scatter(x_train,y_train ,color ='red')
plt.plot(x_train , regressor.predict(x_train),color='blue')
plt.title('Salary vs Experianc(Train set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()