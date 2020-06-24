# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:51:39 2020

@author: Bhuvi
"""

# In this classification example we are trying to classify whether a person has heart disease or not using decision tree algorithm

import pandas as pd # to load and manipulate data and for One-Hot encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw the decision tree
from sklearn.model_selection import train_test_split # to split the data into training and test set
from sklearn.model_selection import cross_val_score # for cross validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix # to draw the confusion matrix


# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = None)

df = pd.read_csv("G:/Python Materail/processed.cleveland.data", header = None)

df.head(5)

df.columns = ['age',
              'sex',
              'cp',
              'restbp',
              'chol',
              'fbs',
              'restecg',
              'thalach',
              'exang',
              'oldpeak',
              'slope',
              'ca',
              'thal',
              'hd']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df.head(5)



# CHECK MISSING VALUE IN THE DATASET

print(df.isnull().any()) # There are no missing values

# CHECK DATA TYPES OF COLUMNS

df.dtypes # Columns 'ca' and 'thal' has to be float

df['ca'].unique
df['thal'].unique

# To get the locations where column 'ca' and 'thal' has '?'

df.loc[(df['ca'] == '?') | (df['thal'] == '?')]


# To get the count of number of observation with '?' in ca and thal column

len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')] )

# Save dataset with out '?'

df_no_missing = df[(df['ca'] != '?') & (df['thal'] != '?')]

len(df_no_missing)

df_no_missing.head(5)

# CHECK FOR '?' IN THE NEW DATASET

df_no_missing['ca'].unique

df_no_missing['thal'].unique

# '?' ARE REMOVED

# DATA TYPE CHECK, MISSING VALUES CHECK DONE

# DATA INDEPENDENT AND OUTCOME VARIABLE SPLIT 

X = df_no_missing.drop('hd', axis = 1).copy()

X.head()

# ALTERNATIVELY : X = df_no_missing.iloc[:,:-1]

y = df_no_missing['hd'].copy()

y.head()


# --------------- ONE HOT ENCODING ------------------------------

# get_dummies() method used for One-Hot Encoding

X_encoded = pd.get_dummies(X, columns = ['cp',
                                         'restecg',
                                         'slope',
                                         'thal'])
# SHOWS THE ENCODED 'cp', 'restecg','slope' ,'thal' COLUMNS

X_encoded.head()

# In this example we are trying to classify only whether a person has heart disease or not
# So we will make group all the catergories of column 'hd' > 0 as 1
# 'hd' equals 0 means no heart disease and 'hd' equals 1 means person has heart disease

y_heart_disease_index = y > 0 # gives the index of all non zero values in y

y[y_heart_disease_index] = 1 # update all the non zero value in y to 1

y.unique()

#------------------------------------------------------------


#---------- Preliminary Classification Tree ------------------

## SPLIT   

X_train, X_test, y_train, y_test = train_test_split( X_encoded, y, random_state = 42 )

# Create decision tree and fit it to the training data

clsf_des_tree = DecisionTreeClassifier(random_state = 42)
clsf_des_tree = clsf_des_tree.fit( X_train, y_train)

## plot the decision tree

plt.figure(figsize = (15, 7.5))
plot_tree(clsf_des_tree,
          filled = True,
          rounded = True,
          class_names = ["No HD", "Yes HD"],
          feature_names = X_encoded.columns);

## Plot confusion matrix 

plot_confusion_matrix(clsf_des_tree, X_test, y_test, display_labels = ["No HD", "Yes HD"])

## pruning the tree to fix over fitting issue

path = clsf_des_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas # extract different value for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value of alpha

clf_dts = []

# Now create one decision tree per value for alpha and store it in the array

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)


train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores =  [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]


fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Alpha for training and test sets")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle ="steps_post")
ax.plot(ccp_alphas, test_scores, marker = 'o', label = "test", drawstyle ="steps_post")
ax.legend()
plt.show()


# --------------------- CROSS VALIDATION -----------------------


clf_dt = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.016)
scores = cross_val_score(clf_dt, X_train, y_train, cv = 5)
df = pd.DataFrame(data = {'tree' : range(5), 'accuracy' : scores})
df.plot(x='tree', y = 'accuracy', marker = 'o', linestyle = '--')


alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv = 5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])


alpha_results = pd.DataFrame(alpha_loop_values,
                             columns =  ['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x = 'alpha',
                   y = 'mean_accuracy',
                   yerr = 'std',
                   marker = 'o',
                   linestyle = '--')

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                                &
                                (alpha_results['alpha'] < 0.015)]['alpha']

ideal_ccp_alpha
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha

#-------------------------------------------------------------  


# Build and train new decision tree with new optimal alpha value

clf_dt_pruned = DecisionTreeClassifier(random_state = 42, ccp_alpha = ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels = ["No HD", "Yes HD"])

plt.figure(figsize = (15, 7.5))
plot_tree(clf_dt_pruned,
          filled = True,
          rounded = True,
          class_names = ["No HD", "Yes HD"],
          feature_names = X_encoded.columns);


