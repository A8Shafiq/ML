# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:30:56 2024

@author: amr.shafek
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


### Read the data using numpay Libray  
data = np.load("mnist_train_small.npy")


# Show the data and we found that its (19999, 785)  ist 19999 record and it has 785 Colunns 
print(data.shape)

# This means x contains all rows of data but excludes the first column, which is often the label or target in a supervised learning dataset.
x = data[:,1:]
# This means y contains all rows of just the first column, which is typically the label or target value for each example in data.
y = data[:,0]

print(x[0])
print(y[0])

# to show the data 
plt.imshow(X[0].reshape(28,28),cmap="gray")


#Split the data into Tain and test with .33 as trainng and 42 is the point to splite or suffle 
X_train,X_test,yTrain,yTest = train_test_split(x,y,test_size=0.33,random_state=42)


#Call our MOdel
Model = KNeighborsClassifier()

# where is our model fit our data and as we know the KNN is do nothing on Fit Part 
#just ploting the points on graph 
Model.fit(X_train,yTrain)

# We predict the first 10 Numbers on our sets 
Model.predict(X_test[:10])
# and we print the first 10 result to see if our model work right 
yTest[:10]

# here our mode give me the first result as 1 and its 7 so i look at it and its look like one XD

plt.imshow(X_test[0].reshape(28,28))

#pritn my score 
Model.score(X_test,yTest)




