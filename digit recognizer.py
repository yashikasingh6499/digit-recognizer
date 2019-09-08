#importing the libraries
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#reading the data
data = pd.read_csv("train.csv").as_matrix()
clf = DecisionTreeClassifier()

#training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

#fitting the data into classifier
clf.fit(xtrain,train_label)

#test dataset
xtest = data[21000:,1:]
actual_label = data[21000:,0]

#predicting and visualization of data
d = xtest[721]
d.shape = (28,28)
pt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[721]]))
pt.show()   

#accuracy of model
p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    count+=1 if p[i] == actual_label[i] else 0
print("Accuracy = ",(count/21000)*100)