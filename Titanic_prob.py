import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
#Reading the train and test dataset
train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#Placing the mean values of ages i the missing rows
avg_age = np.mean(train["Age"])
train["Age"].fillna(avg_age, inplace = True)

avg_age1 = np.mean(test["Age"])
test["Age"].fillna(avg_age1, inplace = True)

sn.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Droping the cabin column
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
print(train.info())
#Using categorical variables in place of male and female
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

sex = pd.get_dummies(test['Sex'],drop_first=True)
embark= pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
train.drop(['Survived'],axis=1,inplace=True)

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

X_train=train.iloc[: ,1:-1].values
y_train=train.iloc[: , 8:9].values
X_test=test.iloc[: ,1:-1].values
y_test=test.iloc[:,8:9].values

#Using RandomForest to train the dataset
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train,y_train)
#Predicting the passenger will survive or not
predictions = clf.predict(X_test)
final = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":predictions})

final.to_csv("finalpred.csv")
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, predictions)
a=accuracy_score(y_test, predictions)
