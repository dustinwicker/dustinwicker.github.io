---
layout: post
title: "Predicting the Presence of Heart Disease in Patients"
author: "Dustin Wicker"
categories: journal
tags: [healthcare,data science,data analytics,data analysis,machine learning,sample]
image: heart.png
---

## Project Summary  
* **Statistical analysis**, **data mining techniques**, and **five machine learning models** were built and ensembled to **accurately predict the presence of heart disease in patients from the Hungarian Institute of Cardiology in Budapest**.  
* The model which provided the optimal combination of total patients predicted correctly and F1 Score, while being the most parsimonious, was the **Support Vector Machine Classification Model #4**. It was able to **correctly predict** the presence, or lack thereof, of heart disease in **86% of patients**.  
  
  
A summary of this models results can be seen directly below, and a full summary of all sixteen models can be found in the [Model Visualization, Comparison, and Selection section](#model-visualization-comparison-and-selection).

| Model                                                                                                                                                   | F1 Score | Recall | Precision | Total Correct | Total Incorrect |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | :------: | :----: | :-------: | :-----------: | :-------------: |
| **Support Vector Machine Classifier Four**                                                                                                                      | **0.804**    | **0.774**  | **0.837**     | **252**           | **40**             |

 
 (Put usefulness up here as well in bullets)
 
Code snippets will be provided for each section outlined in the [Project Overview](#project-overview) at the bottom of this page. If you would like to view the entire code script, please visit this [link](https://github.com/dustinwicker/Heart-Disease-Detection/blob/master/heart_disease_code.py).
 
# Project Overview  
## i.    [Data Ingestion](#data-ingestion)
## ii.   [Data Cleaning](#data-cleaning)
## iii.  [Exploratory Data Analysis](#exploratory-data-analysis)
## iv.  [Model Building](#model-building)
## v.   [Model Visualization, Comparison, and Selection](#model-visualization-comparison-and-selection)
## vi.  [Visualize Best Model](#visualize-best-model)
## vii. [Model Usefulness](#model-usefulness)
  
## Data Ingestion
The first step was obtaining the [data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data) and [data dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names) from the UCI Machine Learning Repository. The files were saved in an appropriate location on my machine and then read into Python. [<sub><sup>View code</sup></sub>](#a)

## Data Cleaning  
After the data was properly read into into Python and the appropriate column names were supplied, data cleaning was performed.[<sub><sup>View code</sup></sub>](#b)  
This involved:
* Removing unnecessary columns
* Converting column types
* Correcting any data discrepancies
* Removing patients with a large percentage of missing values
   * In this particular case, large meant 10% and as a result, two patients were removed from the data set.
* Imputing missing values for patients using K-Nearest Neighbors, an advanced data imputation method  

## Exploratory Data Analysis
The following two images provide a sample of the analysis performed.

![Distribution_of_Continuous_Features_by_Target](/assets/img/distribution_of_continuous_features_by_target.png "Distribution of Continuous Features by Target")

(Link to code snippet at bottom of page for all independent code bodies)

There is a histogram for each of the initial continuous features against the target variable (diagnosis of heart disease). This visualization allows you to see which of the predictor variables have noticeable differences in their distributions when split on the target and would therefore be useful in prediction (a good example of this is "Maximum Heart Rate Achieved.")

![Heatmap of Continous Predictor Variables](/assets/img/heatmap_continous_predictor_variables.png "Heatmap of Continous Predictor Variables")

This heatmap shows correlation coefficients between the initial continuous variables plus two features, "Days Between Cardiac Catheterization and Electrocardiogram" and "PCA variable for 'Height at Rest' and 'Height at Peak Exercise'", created in the early stages of feature enginering. This visualization gives you information that is useful in performing data transformations and (further) feature engineering.  

Including the details above, this step also involved:
* Statistical Analysis  
   * Chi-Square Tests  
   * Fisher's Exact Chi-Square Tests  
   * Odds Ratios
   * Contingency Tables
   * Normality Tests
* Feature Engineering (do visualization will of this done?)
* Data Visualization
* Data Transformations (add section for this? visualizaton showing the differences - chol)

## Model Building
After exploring our data to obtain a greater understanding of it and using that information to perform feature engineering and data transformations, it was time to build and optimize models.
* Five different machine learning algorithms were used  
   * Logistic Regression  
   * Random Forest 
   * K-Nearest Neighbors
   * Support Vector Machine
   * Gradient Boosting
* Seven unique sets of variables were created with each set containing continuous and categorical features
* Each model was run on every set of variables (for a total of 35 models) - bold 35 models
* Every model run was optimized using
   * Grid search  
   * Cross-validation
   * Feature importance techniques
   
## Model Visualization, Comparison, and Selection  
* ROC Curves were built based on each model's predicted probabilities to visually compare model performance at various cut-off values.  
Below the four models which give predicted probabilities (Support Vector Machines do not give predicted probabilities, only class membership) are plotted, and each plot contains seven ROC curves - one for each unique sets of variables. The most amount of variation can be seen in the Random Forest Classifier models, and the least amount in the Logistic Regression models due to the fact variables had to be statistically signifcant to be included in the model.

![ROC Curves](/assets/img/roc_cruves.png "ROC Curves")

* The best model for each algorithm was selected based on a combination of the total patients predicted correctly and F1 Score.
* From there, model predictions were assembled to determine which combination (or stand alone model) provided the best results.
* A summary of the model results can be seen below.
* The best model, the Support Vector Machine Classification Model #4, is bolded. It is the most parsimonious model which provided the optimal combination of total patients predicted correctly and F1 Score.

| Model(s)                                                                                                                                                   | F1 Score | Recall | Precision | Total Correct | Total Incorrect |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | :------: | :----: | :-------: | :-----------: | :-------------: |
| Gradient Boosting Classifer Five, K-Nearest Neighbors Five, Support Vector Machine Classifier Four                                                         | 0.802    | 0.745  | 0.868     | 253           | 39              |
| **Support Vector Machine Classifier Four**                                                                                                                      | **0.804**    | **0.774**  | **0.837**     | **252**           | **40**             |
| Gradient Boosting Classifer Five, Random Forest Classifer Six, Support Vector Machine Classifier Four                                                      | 0.798    | 0.745  | 0.859     | 252           | 40              |
| Gradient Boosting Classifer Five, K-Nearest Neighbors Five, Logistic Regression Three, Random Forest Classifer Six, Support Vector Machine Classifier Four | 0.798    | 0.745  | 0.859     | 252           | 40              |
| Logistic Regression Three, Random Forest Classifer Six, Support Vector Machine Classifier Four                                                             | 0.792    | 0.755  | 0.833     | 250           | 42              |
| K-Nearest Neighbors Five                                                                                                                                   | 0.786    | 0.726  | 0.856     | 250           | 42              |
| K-Nearest Neighbors Five, Random Forest Classifer Six, Support Vector Machine Classifier Four                                                              | 0.786    | 0.726  | 0.856     | 250           | 42              |
| K-Nearest Neighbors Five, Logistic Regression Three, Support Vector Machine Classifier Four                                                                | 0.786    | 0.745  | 0.832     | 249           | 43              |
| Gradient Boosting Classifer Five, K-Nearest Neighbors Five, Logistic Regression Three                                                                      | 0.782    | 0.726  | 0.846     | 249           | 43              |
| Gradient Boosting Classifer Five, Logistic Regression Three, Support Vector Machine Classifier Four                                                        | 0.782    | 0.726  | 0.846     | 249           | 43              |
| Gradient Boosting Classifer Five                                                                                                                           | 0.771    | 0.698  | 0.860     | 248           | 44              |
| Gradient Boosting Classifer Five, Logistic Regression Three, Random Forest Classifer Six                                                                   | 0.772    | 0.717  | 0.835     | 247           | 45              |
| K-Nearest Neighbors Five, Logistic Regression Three, Random Forest Classifer Six                                                                           | 0.772    | 0.717  | 0.835     | 247           | 45              |
| Gradient Boosting Classifer Five, K-Nearest Neighbors Five, Random Forest Classifer Six                                                                    | 0.769    | 0.708  | 0.843     | 247           | 45              |
| Random Forest Classifer Six                                                                                                                                | 0.768    | 0.717  | 0.826     | 246           | 46              |
| Logistic Regression Three                                                                                                                                  | 0.765    | 0.736  | 0.796     | 244           | 48              |

## Visualize Best Model
The next step was visualizing the results of the best model in an easy to understand way. 


 ## Model Usefulness
The final step, and argubably most critical one, is explaining how the results could be utilized in a medical facility setting to benefit medical practitioners and their patients.
* The model could be implemented, along with the patient's formal checkups and examinations, to assist medical practitioners in correctly diagnosing heart disease in their patients.  
* It would allow practitioners to get an in-depth understanding of which factors contribute to heart disease, and set up a prehabilitation routine for their patients that would help decrease those factors (such as helping the patient establish a diet and exercise regimen to decrease their serum cholesterol). This would provide patients a path towards a clean bill of health, and prevent possible heart disease in the future.



# a
```python
# Import libraries and modules
import os
import yaml
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from math import sqrt
from scipy import stats
import statsmodels.api as sm
import inflect
import itertools
from more_itertools import unique_everseen
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Be able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', -1)
# Increase number of rows printed out in console
pd.set_option('display.max_rows', 200)

# Change current working directory to main directory
def main_directory():
    # Load in .yml file to retrieve location of heart disease directory
    info = yaml.load(open("info.yml"), Loader=yaml.FullLoader)
    os.chdir(os.getcwd() + info['heart_disease_directory'])
main_directory()

# Open Hungarian data set
with open('hungarian.data', 'r') as myfile:
    file = []
    for line in myfile:
        line = line.replace(" ", ", ")
        # Add comma to end of each line
        line = line.replace(os.linesep, ',' + os.linesep)
        line = line.split(', ')
        file.extend(line)

file = [value.replace(",\n", "") for value in file]
# Remove empty strings from list
file = list(filter(None, file))

# Convert list to lists of list
attributes_per_patient = 76 # len(file)/number of patients
i = 0
new_file = []
while i < len(file):
    new_file.append(file[i:i+attributes_per_patient])
    i += attributes_per_patient

# List of column names
headers = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol',
           'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop',
           'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps',
           'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm',
           'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo',
           'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox',
           'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']

# Convert lists of list into DataFrame and supply column names
hungarian = pd.DataFrame(new_file, columns=headers)
```
  
# b
* Remove unnecessary columns
```python
# List of columns to drop
cols_to_drop =['ccf', 'pncaden', 'smoke', 'cigs', 'years', 'dm', 'famhist', 'dig', 'ca', 'restckm', 'exerckm',
               'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'lmt',
               'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1',
               'cathef', 'junk', 'name', 'thaltime', 'xhypo', 'slope', 'dummy', 'lvx1', 'lvx2']

# Drop columns from above list
hungarian = hungarian.drop(columns=cols_to_drop)
```

* Convert column types
```python
# Convert all columns to numeric
hungarian = hungarian.apply(pd.to_numeric)
```

* Correct data discrepancies
```python
### Fix possible patient id issues
# Find ids that are not unique to patients
print(hungarian.id.value_counts()[hungarian.id.value_counts()!=1])

# Fix id 1132 (two different patients are both assigned to this id) - give second patient next id number (id max + 1)
hungarian.loc[hungarian.loc[hungarian.id==1132].index[-1], 'id'] = hungarian.id.max() + 1
```
* Remove patients with a large percentage of missing values
```python
# Drop patients with "significant" number of missing values (use 10%, can adjust accordingly)
# Determine missing value percentage per patient (-9 is the missing attribute value)
missing_value_perc_per_patient = (hungarian == -9).sum(axis=1)[(hungarian == -9).sum(axis=1) > 0]\
                                     .sort_values(ascending=False)/len([x for x in hungarian.columns if x != 'id'])

# Remove patients with > 10% missing values
hungarian = hungarian.drop(missing_value_perc_per_patient[missing_value_perc_per_patient>0.10].index.values)
```
* Impute missing values
```python
### Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x: x[1])

# Use K-Nearest Neighbors (KNN) to impute missing values
# Method to scale continuous and binary variables (z-score standardization)
scaler = StandardScaler()
variables_not_to_use_for_imputation = ['ekgday', 'cmo', 'cyr', 'ekgyr', 'cday', 'ekgmo', 'num']

# Impute htn
impute_variable = 'htn'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use
fix_htn = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'nitr', 'pro', 'diuretic', 'exang',
                           'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_htn[value], prefix=value)
    fix_htn = fix_htn.join(one_hot)
    fix_htn = fix_htn.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_htn) if x != impute_variable]

# Create DataFrame with missing value(s) to predict on
predict = fix_htn.loc[fix_htn[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_htn.loc[~(fix_htn[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit and transform scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
htn_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for htn is {htn_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, 'htn'] = htn_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute restecg
impute_variable = 'restecg'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - added in 'htn'
fix_restecg = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'nitr', 'pro', 'diuretic', 'exang',
                           'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_restecg[value], prefix=value)
    fix_restecg = fix_restecg.join(one_hot)
    fix_restecg = fix_restecg.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_restecg) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_restecg.loc[fix_restecg[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_restecg.loc[~(fix_restecg[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit and transform scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
restecg_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for restecg is {restecg_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, 'restecg'] = restecg_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute prop
# Set y variable
impute_variable = 'prop'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'htn'
fix_prop = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'nitr', 'pro', 'diuretic', 'exang',
                           'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_prop[value], prefix=value)
    fix_prop = fix_prop.join(one_hot)
    fix_prop = fix_prop.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_prop) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_prop.loc[fix_prop[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_prop.loc[~(fix_prop[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit and transform scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
prop_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for prop is {prop_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, 'prop'] = prop_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute thaldur
# Set y variable
impute_variable = 'thaldur'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'prop'
fix_thaldur = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'exang', 'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_thaldur[value], prefix=value)
    fix_thaldur = fix_thaldur.join(one_hot)
    fix_thaldur = fix_thaldur.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_thaldur) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_thaldur.loc[fix_thaldur[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_thaldur.loc[~(fix_thaldur[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
thaldur_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The prediction for thaldur is " + str(thaldur_prediction[0]) + ".")
# Round thaldur_prediction to integer
thaldur_prediction = round(number=thaldur_prediction[0])
print("The prediction for thaldur has been rounded to " + str(thaldur_prediction) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, 'thaldur'] = thaldur_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute rldv5
# Set y variable
impute_variable = 'rldv5'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'prop'
fix_rldv5 = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'exang', 'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_rldv5[value], prefix=value)
    fix_rldv5 = fix_rldv5.join(one_hot)
    fix_rldv5 = fix_rldv5.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_rldv5) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_rldv5.loc[fix_rldv5[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_rldv5.loc[~(fix_rldv5[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
rldv5_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The prediction for rldv5 is " + str(rldv5_prediction[0]) + ".")
# Round rldv5_prediction to integer
rldv5_prediction = round(number=rldv5_prediction[0])
print("The prediction for rldv5 has been rounded to " + str(rldv5_prediction) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, 'rldv5'] = rldv5_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute met
# Set y variable
impute_variable = 'met'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'rldv5'
fix_met = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'exang', 'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_met[value], prefix=value)
    fix_met = fix_met.join(one_hot)
    fix_met = fix_met.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_met) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_met.loc[fix_met[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_met.loc[~(fix_met[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
met_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for met are:")
print(met_prediction)

# Round met_prediction to integer
for i in range(len(met_prediction)):
    met_prediction[i] = round(number=met_prediction[i])
    print("The prediction for met_prediction" + "[" + str(i) + "]" + " has been rounded to " + str(met_prediction[i]) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, impute_variable] = met_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute fbs
# Set y variable
impute_variable = 'fbs'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'met'
fix_fbs = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'exang', 'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_fbs[value], prefix=value)
    fix_fbs = fix_fbs.join(one_hot)
    fix_fbs = fix_fbs.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_fbs) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_fbs.loc[fix_fbs[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_fbs.loc[~(fix_fbs[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
fbs_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for fbs are:")
print(fbs_prediction)

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, impute_variable] = fbs_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute fbs
# Set y variable
impute_variable = 'proto'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'fbs'
fix_proto = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'exang', 'lvx3', 'lvx4', 'lvf']

# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_proto[value], prefix=value)
    fix_proto = fix_proto.join(one_hot)
    fix_proto = fix_proto.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_proto) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_proto.loc[fix_proto[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_proto.loc[~(fix_proto[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
proto_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for proto are:")
print(proto_prediction)

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, impute_variable] = proto_prediction

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = [(col, hungarian[col].value_counts()[-9]) for col in list(hungarian) if -9 in hungarian[col].unique()]
# Sort tuples by number of missing values
cols_with_missing_values.sort(key=lambda x:x[1])

# Impute chol
impute_variable = 'chol'

# Obtain list of variables to use for imputation
x_variables = [x for x in list(hungarian) if x not in [x[0] for x in cols_with_missing_values] +
                        variables_not_to_use_for_imputation + ['id']]

# Select x and y variables to use - add in 'fbs'
fix_chol = hungarian[x_variables + [impute_variable]]

# Create list of categorical variables to one-hot encode
categorical_x_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop', 'nitr', 'pro',
                           'diuretic', 'proto', 'exang', 'lvx3', 'lvx4', 'lvf']


# One-hot encode categorical variables
for value in categorical_x_variables:
    one_hot = pd.get_dummies(fix_chol[value], prefix=value)
    fix_chol = fix_chol.join(one_hot)
    fix_chol = fix_chol.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_chol) if x != impute_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_chol.loc[fix_chol[impute_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[impute_variable]

# Create DataFrame to train on
train = fix_chol.loc[~(fix_chol[impute_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[impute_variable]

# Fit scaler on train_x
train_x = scaler.fit_transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print(f"k is {k}.")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")
    # Substract one to make k odd number
    k -= 1
    print(f"k is now {k}.")

# Predict value for predict_y
chol_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for chol are:")
print(chol_prediction)

# Round chol_prediction to integer
for i in range(0, len(chol_prediction)):
    chol_prediction[i] = round(number=chol_prediction[i])
    print(f"The prediction for chol_prediction [{str(i)}] has been rounded to {chol_prediction[i]}.")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[impute_variable]==-9, impute_variable] = chol_prediction
```

(Put code at bottom - base off table of contents and say for all code (script) - go to the Github page for the project (give link to heart disease))

Questions: how else could the model be useful? Other visuals that could be useful to visualize the results? Any strategies that could make the model more useful? Feature engineering ideas? Any questions or lack of understanding on code?


Hi there! I'm Paul. I’m a physics major turned programmer. Ever since I first learned how to program while taking a scientific computing for physics course, I have pursued programming as a passion, and as a career. Check out [my personal website](https://www.lenpaul.com/) for more information on my other projects (including more Jekyll themes!), as well as some of my writing.
