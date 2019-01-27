import numpy as np
import pandas as pd

dataset = pd.read_csv('yeah.csv')
data = dataset.iloc[:,1:9]

X = data.iloc[:,0:7]
Y= data.iloc[:,-1]

X= np.asarray(X)
Y= np.asarray(Y)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred = y_pred *100
y_pred = y_pred.round()
y_pred = y_pred/100


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt

a= []
for i in range(0,100):
    i = i/100
    a.append(i)
    
    
plt.plot(a,y_pred)
plt.plot(a,y_test)
plt.show()    