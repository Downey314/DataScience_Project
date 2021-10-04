#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:56:27 2021

@author: downey
"""

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA

from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Standardizing
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sn



# for i in ('gbk','utf-8','gb18030','ansi'): 
#     try:
#         voice = pd.read_csv('/Users/downey/Desktop/data1.csv',encoding = i,  header=1)
#         print(i + 'decode success')
#     except:
#         print(i + 'decode fail')
        
#Read into the data      
voice = pd.read_csv("/Users/downey/Desktop/DDD.csv")
print(voice.shape)
print(voice.head(10))
        
#Data Types
types = voice.dtypes #Data Types
print(types)

#Null Values
nulls = voice.isnull().sum()
print(nulls)


print(voice.info())
#Distribution of target variable
print(voice["Healthy"].value_counts()) 

#grouping the data based on the target variable
print(voice.groupby("Healthy").mean())

voice = voice.drop(['ID','Age'], axis='columns')

X = voice.drop(['Healthy'], axis='columns')
y = voice.Healthy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)





voice1 = voice[voice.Healthy == 1]
voice2 = voice[voice.Healthy == 2]

plt.scatter(voice1.age_group, voice1['ah_LCS'], color = 'green')
plt.scatter(voice2.age_group, voice2['ah_LCS'], color='red')
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age_group')
plt.ylabel('ah_LCS')
plt.legend()


voice1 = voice[voice.Healthy == 1]
voice2 = voice[voice.Healthy == 2]
plt.scatter(voice1.age_group, voice1['ah_LMX'], color = 'green')
plt.scatter(voice2.age_group, voice2['ah_LMX'], color='red')
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age_group')
plt.ylabel('ah_LMX')
plt.legend()

voice1 = voice[voice.Healthy == 1]
voice2 = voice[voice.Healthy == 2]
plt.scatter(voice1.age_group, voice1['ah_MPT'], color = 'green')
plt.scatter(voice2.age_group, voice2['ah_MPT'], color='red')
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age_group')
plt.ylabel('ah_MPT')
plt.legend()

voice1 = voice[voice.Smoke == 1]
voice2 = voice[voice.Smoke == 2]
plt.scatter(voice1.age_group, voice1['ah_LMX'], color = 'green')
plt.scatter(voice2.age_group, voice2['ah_LMX'], color='red', alpha = 0.3)
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age_group')
plt.ylabel('ah_LMX')
plt.legend()




#Standardizing
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

print(X_train_s.shape)
print(X_test_s.shape)



#Build Support Vector Machine
model = SVC()
model.fit(X_train_s, y_train)
print(model.score(X_test_s, y_test)) #0.53
# accuracy score on training data
X_train_prediction = model.predict(X_train_s)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# accuracy score on training data
X_test_prediction = model.predict(X_test_s)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


plot_confusion_matrix(model,
                      X_test_s,
                      y_test,
                      values_format = "d",
                      display_labels = ["Disease", "Healthy"])

#Build Predictive System
input_data_H = (1.5,3,2,178,1,1,2,2,1,0,0,0,0,0,0,100,3,5,0,0,0,75,77.6,18.36,65.6,86.6,67.4,86.6,98.5,393.75,29.96,87.17,244.58,231.5,4.32,252.67,141.08,393.75,57.17,0.23,4.19,18,4.02,366,25.1,79.18,137.47,125.88,7.94,288.75,77.25,366,52.61,0.38,1.32,27,4.99,392.37,29.06,82.89,140.55,129.72,7.71,314.95,77.41,392.37,49.79,0.35,0.66,28,4.83,125.88,288.75,77.25,366,52.61,0.38,27,4.99)
input_data_D = (1,2,1,157,2,2,2,2,0,12,7,0,2,18.75,4,92.5,8,5,0,2,1,60.4,84.4,12.5,73.1,79.9,62.6,72,86.3,326.5,46.5,82.2,246.6,228.6,4.6,251.6,114.5,366.1,68,0.3,	2.6,21,6.55,367.5,25.8,80.98,180.44,173.81,5.75,277.42,90.08,367.5,37.76,0.21,2.59,24,3.23,368.94,25.82,88.54,199.79,189.1,5.29,290.81,78.13,368.94,47.24,0.24,1.42,27,4.01,173.81,277.42,90.08,367.5,37.76,0.21,24,3.23)
# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_H)
print(input_data_as_numpy_array.shape) #(79, )

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 2):
  print("The Person is healthy")
else:
  print("The Person is not healthy")





param_grid = [
    {"C": [0.5, 1, 10, 100],
     "gamma": ["scale", 1, 0.1, 0.01, 0.001, 0.0001],
     "kernel": ["rbf", "linear", "sigmoid"]
     }
    ]
optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv = 3,
    scoring = "accuracy"
    )
optimal_params.fit(X_train_s, y_train)
print(optimal_params.best_params_)

model = SVC(C = 0.5, gamma = 0.001, kernel = "linear")
model.fit(X_train_s, y_train)
print(model.score(X_test_s, y_test)) #0.8

# print(model.predict([[1.5,3,2,178,1,1,2,2,1,0,0,0,0,0,0,100,3,5,0,0,0,75,77.6,18.36,65.6,86.6,67.4,86.6,98.5,393.75,29.96,87.17,244.58,231.5,4.32,252.67,141.08,393.75,57.17,0.23,4.19,18,4.02,366,25.1,79.18,137.47,125.88,7.94,288.75,77.25,366,52.61,0.38,1.32,27,4.99,392.37,29.06,82.89,140.55,129.72,7.71,314.95,77.41,392.37,49.79,0.35,0.66,28,4.83,125.88,288.75,77.25,366,52.61,0.38,27,4.99]]))
# print(model.predict([[1,2,1,157,2,2,2,2,0,12,7,0,2,18.75,4,92.5,8,5,0,2,1,60.4,84.4,12.5,73.1,79.9,62.6,72,86.3,326.5,46.5,82.2,246.6,228.6,4.6,251.6,114.5,366.1,68,0.3,	2.6,21,6.55,367.5,25.8,80.98,180.44,173.81,5.75,277.42,90.08,367.5,37.76,0.21,2.59,24,3.23,368.94,25.82,88.54,199.79,189.1,5.29,290.81,78.13,368.94,47.24,0.24,1.42,27,4.01,173.81,277.42,90.08,367.5,37.76,0.21,24,3.23]]))
# print(model.predict([[0,2,2,175,1,2,2,2,1,12,9,1,2,9.38,4,90,7,5,0,1,2,75.1,75.3,23.1,71.4,75.3,68.1,70.7,151.25,394.26,36.52,84.39,269.87,259.06,3.86,243.02,151.25,394.26,55.24,0.2,1.98,16,3.53,346.4,35,72.72,117.48,110.58,9.04,298.77,83.92,382.68,41.13,0.35,1.76,27,4.24,359.4,36.51,76.2,116.06,110.8,9.02,317.6,73.5,391.1,34.88,0.3,1.86,29,3.69,110.58,298.77,83.92,382.68,41.13,0.35,27,4.24]]))

#Cross-Validation with SVM
score_SVM = cross_val_score(SVC(gamma='auto'), X, y,cv=3)
print(score_SVM)
print(np.average(score_SVM))




#Build RandomForest Algorithm 
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)   
print(model.score(X_test, y_test))
print(model.predict([[1.5,3,2,178,1,1,2,2,1,0,0,0,0,0,0,100,3,5,0,0,0,75,77.6,18.36,65.6,86.6,67.4,86.6,98.5,393.75,29.96,87.17,244.58,231.5,4.32,252.67,141.08,393.75,57.17,0.23,4.19,18,4.02,366,25.1,79.18,137.47,125.88,7.94,288.75,77.25,366,52.61,0.38,1.32,27,4.99,392.37,29.06,82.89,140.55,129.72,7.71,314.95,77.41,392.37,49.79,0.35,0.66,28,4.83,125.88,288.75,77.25,366,52.61,0.38,27,4.99]]))
print(model.predict([[1,2,1,157,2,2,2,2,0,12,7,0,2,18.75,4,92.5,8,5,0,2,1,60.4,84.4,12.5,73.1,79.9,62.6,72,86.3,326.5,46.5,82.2,246.6,228.6,4.6,251.6,114.5,366.1,68,0.3,	2.6,21,6.55,367.5,25.8,80.98,180.44,173.81,5.75,277.42,90.08,367.5,37.76,0.21,2.59,24,3.23,368.94,25.82,88.54,199.79,189.1,5.29,290.81,78.13,368.94,47.24,0.24,1.42,27,4.01,173.81,277.42,90.08,367.5,37.76,0.21,24,3.23]]))
print(model.predict([[0,2,2,175,1,2,2,2,1,12,9,1,2,9.38,4,90,7,5,0,1,2,75.1,75.3,23.1,71.4,75.3,68.1,70.7,151.25,394.26,36.52,84.39,269.87,259.06,3.86,243.02,151.25,394.26,55.24,0.2,1.98,16,3.53,346.4,35,72.72,117.48,110.58,9.04,298.77,83.92,382.68,41.13,0.35,1.76,27,4.24,359.4,36.51,76.2,116.06,110.8,9.02,317.6,73.5,391.1,34.88,0.3,1.86,29,3.69,110.58,298.77,83.92,382.68,41.13,0.35,27,4.24]]))
y_predicted = model.predict(X_test)


cm = confusion_matrix(y_test, y_predicted)
print(cm)

#%matplotlib inline
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

#Cross-Validation for RandomForest
score_RF = cross_val_score(RandomForestClassifier(n_estimators=30),X, y, cv=3)
print(score_RF)
print(np.average(score_RF))




#Model selection with different parameters
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [0.5, 1, 10, 100],
            'kernel': ["rbf","linear", "sigmoid"]
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1, 5, 10, 20]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)



#LogisticRegression
model = LogisticRegression(C = 1)
model.fit(X_train, y_train)
print(model.score(X_test,y_test))










