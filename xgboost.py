import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset1=pd.read_csv("train.csv")
dataset2=pd.read_csv("test.csv")
y1=dataset1["breed_category"]
y2=dataset1["pet_category"]
dataset1.drop(["breed_category","pet_category"],axis=1,inplace=True)
dataset=pd.concat((dataset1,dataset2)).reset_index(drop=True)
dataset.drop(["pet_id"],axis=1,inplace=True)
#print(dataset["condition"].value_counts())
dataset['condition'].fillna(-1,inplace=True)
dataset['issue_date']=pd.to_datetime(dataset['issue_date'])
dataset['listing_date']=pd.to_datetime(dataset['listing_date'])

x=[]
for d in dataset['issue_date']:
    x.append(d.month)
dataset['issue_month']=x

x=[]
for d in dataset['listing_date']:
    x.append(d.month)    
dataset['listing_month']=x

x=[]
for d in dataset['issue_date']:
   x.append(d.year+(d.month/12.0)+(d.day/365.0)) 
dataset['issue_date']=x

x=[]
for d in dataset['listing_date']:
    x.append(d.year+(d.month/12.0)+(d.day/365.0))
dataset['listing_date']=x
dataset['time']=abs(dataset['listing_date']-dataset['issue_date'])

dataset.drop(['listing_date','issue_date'],axis=1,inplace=True)
dataset['color_type']=pd.get_dummies(dataset['color_type'])
train=dataset.iloc[:,:].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
dataset=sc_x.fit_transform(dataset)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(dataset[:18834],y1,test_size=0.2)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(xtrain,ytrain)

ypredict=classifier.predict(xtest)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypredict))