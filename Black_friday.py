# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:31:28 2018
@author: YashJain
"""

import pandas as pd
import numpy as np

train_all = pd.read_csv("E:\DA-BA\Datasets\Black Friday\\train_all.csv")
train3 = pd.read_csv("E:\DA-BA\Datasets\Black Friday\\train3.csv")
train23 = pd.read_csv("E:\DA-BA\Datasets\Black Friday\\train23.csv")

train_all.info()
train_all.describe()
#train_all['Age'] = str(train_all['Age'])
#train_all['Occupation'] = str(train_all['Occupation'])
#train_all['Marital_Status'] = str(train_all['Marital_Status'])
#train_all['Product_Category_1'] = str(train_all['Product_Category_1'])
#train_all['Product_Category_2'] = str(train_all['Product_Category_2'])
#train_all['Product_Category_3'] = str(train_all['Product_Category_3'])
del train_all['User_ID']
del train_all['Product_ID']
y_all = train_all['Purchase']
del train_all['Purchase']
y_all_cat = np.array([])
y_all.describe()
y_all_ls = []
for i in y_all:
    if i>0 and i<500:
        cat = "0.5"
    elif i>499 and i<1000:
        cat = "1"
    elif i>999 and i<1500:
        cat = "1.5"
    elif i>1499 and i<2000:
        cat = "2"
    elif i>1999 and i<2500:
        cat = "2.5"
    elif i>2499 and i<3000:
        cat = "3"
    elif i>2999 and i<3500:
        cat = "3.5"
    elif i>3499 and i<4000:
        cat = "4"
    elif i>3999 and i<4500:
        cat = "4.5"
    elif i>4499 and i<5000:
        cat = "5"
    elif i>4999 and i<5500:
        cat = "5.5"
    elif i>5499 and i<6000:
        cat = "6"
    elif i>5999 and i<6500:
        cat = "6.5"
    elif i>6499 and i<7000:
        cat = "7"
    elif i>6999 and i<7500:
        cat = "7.5"
    elif i>7499 and i<8000:
        cat = "8"
    elif i>7999 and i<8500:
        cat = "8.5"
    elif i>8499 and i<9000:
        cat = "9"
    elif i>8999 and i<9500:
        cat = "9.5"
    elif i>9499 and i<10000:
        cat = "10"
    elif i>9999 and i<10500:
        cat = "10.5"
    elif i>10499 and i<11000:
        cat = "11"
    elif i>10999 and i<11500:
        cat = "11.5"
    elif i>11499 and i<12000:
        cat = "12"
    elif i>11999 and i<12500:
        cat = "12.5"
    elif i>12499 and i<13000:
        cat = "13"
    elif i>12999 and i<13500:
        cat = "13.5"
    elif i>13499 and i<14000:
        cat = "14"
    elif i>13999 and i<14500:
        cat = "14.5"
    elif i>14499 and i<15000:
        cat = "15"
    elif i>14999 and i<15500:
        cat = "15.5"
    elif i>15499 and i<16000:
        cat = "16"
    elif i>15999 and i<16500:
        cat = "16.5"
    elif i>16499 and i<17000:
        cat = "17"
    elif i>16999 and i<17500:
        cat = "17.5"
    elif i>17499 and i<18000:
        cat = "18"
    elif i>17999 and i<18500:
        cat = "18.5"
    elif i>18499 and i<19000:
        cat = "19"
    elif i>18999 and i<19500:
        cat = "19.5"
    elif i>19499 and i<20000:
        cat = "20"
    elif i>19999 and i<20500:
        cat = "20.5"
    elif i>20499 and i<21000:
        cat = "21"
    elif i>20999 and i<21500:
        cat = "21.5"
    elif i>21499 and i<22000:
        cat = "22"
    elif i>21999 and i<22500:
        cat = "22.5"
    elif i>22499 and i<23000:
        cat = "23"
    elif i>22499 and i<23500:
        cat = "23.5"      
    elif i>23999 and i<24000:
        cat = "24"      
    else:
        cat = "other"
    y_all_ls.append(cat)
y_all_ls.info()
y_all_ls.Value()
y_all_ls_df = pd.DataFrame(y_all_ls)
y_all_ls_df.columns[0]= ['Purchase']
train_all_train.columns
#train_all = pd.get_dummies(train_all,drop_first = True)
train_all = pd.get_dummies(train_all,columns = ['Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3'],drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#train_all_train, train_all_test, y_all_train, y_all_test = train_test_split(train_all, y_all, test_size = 0.25, random_state = 0)
train_all_train, train_all_test, y_all_train, y_all_test = train_test_split(train_all, y_all_ls_df, test_size = 0.25, random_state = 0)
train_all_train.info()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
train_all_train.columns
#get_dummies
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(train_all_train, y_all_train)
y_all_test = np.reshape(y_all_test, (np.product(y_all_test.shape),1))
y_all_test_predict = classifier.predict(train_all_test)

#Fitting LinearRegression
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((166821, 1)).astype(int), values = train_all, axis = 1)
X_opt = X[:, :]
regressor_OLS.summary()

regressor_OLS = sm.OLS(endog = y_all, exog = train_all).fit()

for i in range(0,5):
    regressor_OLS = sm.OLS(endog = y_all, exog = train_all).fit()
    list_index = []
    list_pval = []
    for i in range(1,len(regressor_OLS.pvalues.index)-1):
        list_index.append(regressor_OLS.pvalues.index[i])
        list_pval.append(regressor_OLS.pvalues[i])
    str1=str(regressor_OLS.pvalues.index[list_pval.index(max(list_pval))+1])
    del train_all[str1]

from sklearn.cross_validation import train_test_split
#train_all_train, train_all_test, y_all_train, y_all_test = train_test_split(train_all, y_all, test_size = 0.25, random_state = 0)
train_all_train, train_all_test, y_all_train, y_all_test = train_test_split(train_all, y_all, test_size = 0.25, random_state = 0)
y_all_train = y_all_train.astype(np.float64)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_all_train,y_all_train)
# Predicting the Test set results
y_pred = regressor.predict(train_all_test)
from sklearn.metrics import accuracy_score,r2_score
r2_score(y_all_test,y_pred)

#checking accuracy
from sklearn.metrics import accuracy_score,r2_score
accuracy_score(y_all_test,y_all_test_predict)

