# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:08:01 2022

@author: ashwi
"""
import pandas as pd
SD_train = pd.read_csv("SalaryData_Train.csv")

SD_train.shape
SD_train.dtypes
SD_train.head()
SD_train["workclass"].value_counts()
SD_train["education"].value_counts()
SD_train["maritalstatus"].value_counts()
SD_train["capitalgain"].value_counts()
SD_train["native"].value_counts()
SD_train["Salary"].value_counts()
SD_train["hoursperweek"].value_counts()

#=========================================================================
# blankes
SD_train.isnull().sum()

# finding duplicate row
SD_train.duplicated()
SD_train[SD_train.duplicated()].head()
SD_train.duplicated().sum()

SD_train = SD_train.drop_duplicates()

# finding duplicate columns
SD_train.columns.duplicated()
SD_train.columns.duplicated().sum()


SD_train.isna()
SD_train.isna().sum()
#=========================================================================
# Data visualization

import matplotlib.pyplot as plt

def plot_boxplot(SD_train,ft):
    SD_train.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

plot_boxplot(SD_train,"age")
plot_boxplot(SD_train,"educationno")
plot_boxplot(SD_train,"capitalgain")
plot_boxplot(SD_train,"capitalloss")
plot_boxplot(SD_train,"hoursperweek")


def outliers(SD_train,ft):
    Q1 = SD_train[ft].quantile(0.25)
    Q3 = SD_train[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 -1.5*IQR
    upper_bound = Q3 +1.5*IQR
    ls = SD_train.index[(SD_train[ft] < lower_bound) | (SD_train[ft] > upper_bound)]
    return ls

index_list = []
for feature in ["age","educationno", "capitalgain", "capitalloss", "hoursperweek"]:
    index_list.extend(outliers(SD_train,feature))

index_list

def remove(SD_train,ls):
    ls = sorted(set(ls))
    SD_train = SD_train.drop(ls)
    return SD_train

SD_train_cleaned = remove(SD_train,index_list)
SD_train_cleaned
#===========================================================
# histogram
SD_train_cleaned["age"].hist()
SD_train_cleaned["educationno"].hist()
SD_train_cleaned["capitalgain"].hist()
SD_train_cleaned["capitalloss"].hist()
SD_train_cleaned["hoursperweek"].hist()
#===================================================================
# scatter plot
import seaborn as sns
sns.pairplot(SD_train_cleaned)
#===============================================================
SD_train_cleaned.dtypes

# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
SD_train_cleaned["workclass"] = LE.fit_transform(SD_train_cleaned["workclass"])
SD_train_cleaned["education"] = LE.fit_transform(SD_train_cleaned["education"])
SD_train_cleaned["maritalstatus"] = LE.fit_transform(SD_train_cleaned["maritalstatus"])
SD_train_cleaned["occupation"] = LE.fit_transform(SD_train_cleaned["occupation"])
SD_train_cleaned["relationship"] = LE.fit_transform(SD_train_cleaned["relationship"])
SD_train_cleaned["race"] = LE.fit_transform(SD_train_cleaned["race"])
SD_train_cleaned["sex"] = LE.fit_transform(SD_train_cleaned["sex"])
SD_train_cleaned["native"] = LE.fit_transform(SD_train_cleaned["native"])
SD_train_cleaned["Salary"] = LE.fit_transform(SD_train_cleaned["Salary"])

SD_train_cleaned.dtypes

#=============================================================
X = SD_train_cleaned.iloc[:,0:13]
Y =SD_train_cleaned.iloc[:,13]

#======================================================
# model fitting
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X,Y)
Y_pred = NB.predict(X)

from sklearn.metrics import accuracy_score
print("Accuracy Score :", accuracy_score(Y, Y_pred))

from sklearn.metrics import confusion_matrix,log_loss
cm = confusion_matrix(Y, Y_pred)
cm

log_loss(Y,Y_pred).round(3)
#========================================================

# Regularization
# ridge classifier
from sklearn.linear_model import RidgeClassifier
Rg = RidgeClassifier(alpha=500)
Rg.fit(X,Y)
Y_pred = Rg.predict(X)

from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(Y, Y_pred))

# Lasso classifior
from sklearn.linear_model import Lasso
LaR = Lasso(alpha =1)
LaR.fit(X,Y)
Y_Pred = LaR.predict(X)

print("Accuracy :",accuracy_score(Y, Y_pred))

pd.DataFrame(LaR.coef_)
pd.DataFrame(X.columns)

d1 = pd.concat([pd.DataFrame(LaR.coef_),pd.DataFrame(X.columns)],axis=1)
d1

##############################################################################
##############################################################################

import pandas as pd
SD_test = pd.read_csv("SalaryData_test.csv")

SD_test.shape
SD_test.dtypes
SD_test.head()
SD_test["workclass"].value_counts()
SD_test["education"].value_counts()
SD_test["maritalstatus"].value_counts()
SD_test["capitalgain"].value_counts()
SD_test["native"].value_counts()
SD_test["Salary"].value_counts()
SD_test["hoursperweek"].value_counts()
#============================================================================

# blankes
SD_test.isnull().sum()

# finding duplicate row
SD_test.duplicated()
SD_test[SD_test.duplicated()].head()
SD_test.duplicated().sum()

SD_test = SD_test.drop_duplicates()
SD_test
# finding duplicate columns
SD_test.columns.duplicated()
SD_test.columns.duplicated().sum()


SD_test.isna()
SD_test.isna().sum()
#==============================================================================
# Data visualization for SD_test

import matplotlib.pyplot as plt

def plot_boxplot(SD_test,ft):
    SD_test.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

plot_boxplot(SD_test,"age")
plot_boxplot(SD_test,"educationno")
plot_boxplot(SD_test,"capitalgain")
plot_boxplot(SD_test,"capitalloss")
plot_boxplot(SD_test,"hoursperweek")


def outliers(SD_test,ft):
    Q1 = SD_test[ft].quantile(0.25)
    Q3 = SD_test[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 -1.5*IQR
    upper_bound = Q3 +1.5*IQR
    ls = SD_test.index[(SD_test[ft] < lower_bound) | (SD_test[ft] > upper_bound)]
    return ls

index_list = []
for feature in ["age","educationno", "capitalgain", "capitalloss", "hoursperweek"]:
    index_list.extend(outliers(SD_test,feature))

index_list

def remove(SD_test,ls):
    ls = sorted(set(ls))
    SD_test = SD_test.drop(ls)
    return SD_test

SD_test_cleaned = remove(SD_test,index_list)
SD_test_cleaned
#================================================================================
# satter plot
import seaborn as sns
sns.pairplot(SD_test_cleaned)

#=============================================================================
# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
SD_test_cleaned["workclass"] = LE.fit_transform(SD_test_cleaned["workclass"])
SD_test_cleaned["education"] = LE.fit_transform(SD_test_cleaned["education"])
SD_test_cleaned["maritalstatus"] = LE.fit_transform(SD_test_cleaned["maritalstatus"])
SD_test_cleaned["occupation"] = LE.fit_transform(SD_test_cleaned["occupation"])
SD_test_cleaned["relationship"] = LE.fit_transform(SD_test_cleaned["relationship"])
SD_test_cleaned["race"] = LE.fit_transform(SD_test_cleaned["race"])
SD_test_cleaned["sex"] = LE.fit_transform(SD_test_cleaned["sex"])
SD_test_cleaned["native"] = LE.fit_transform(SD_test_cleaned["native"])
SD_test_cleaned["Salary"] = LE.fit_transform(SD_test_cleaned["Salary"])

SD_test_cleaned.dtypes

#==========================================================================
X1 = SD_test_cleaned.iloc[:,0:13]
Y1 =SD_test_cleaned.iloc[:,13]
#======================================================================

# model fitting
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X1,Y1)
Y1_pred = NB.predict(X1)

from sklearn.metrics import accuracy_score
print("Accuracy Score :", accuracy_score(Y1, Y1_pred))

from sklearn.metrics import confusion_matrix,log_loss
cm = confusion_matrix(Y1, Y1_pred)
cm

log_loss(Y1,Y1_pred).round(3)





















