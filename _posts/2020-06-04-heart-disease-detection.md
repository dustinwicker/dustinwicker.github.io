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
* Setting the target variable ("num") to a binary range as previous studies have done

## Exploratory Data Analysis 
The following three images provide a sample of the analysis performed.

![Heatmap of Continous Predictor Variables](/assets/img/heatmap_continous_predictor_variables.png "Heatmap of Continous Predictor Variables")

This heatmap shows correlation coefficients between the initial continuous variables plus two features, "Days Between Cardiac Catheterization and Electrocardiogram" and "PCA variable for 'Height at Rest' and 'Height at Peak Exercise'", created in the early stages of feature enginering. This visualization gives you information that is useful in performing data transformations and (further) feature engineering. 

![Distribution_of_Continuous_Features_by_Target](/assets/img/distribution_of_continuous_features_by_target.png "Distributions of Continuous Features by Target")

There is a histogram for each of the initial continuous features against the target variable (diagnosis of heart disease). This visualization allows you to see which of the predictor variables have noticeable differences in their distributions when split on the target and would therefore be useful in prediction  
* A good example of this is "Maximum Heart Rate Achieved."

![Serum_Cholesterol_Distribution_with_KDE_Overlaid](/assets/img/chol_data_transformation.png "Serum Cholesterol Distribution with KDE Overlaid")

The above histograms show Serum Cholesterol on the left with no data transformation performed. Notice the high, positive kurtosis value (please note this value is an adjusted verison of Pearson's kurtosis, known as the excess kurtosis, where three is subtracted from the original kurtosis value to provide the comparison to a normal distribution - a value of 0.0 would be the excess kurtosis value of a normal distribution). This is a lepotkurtotic distribution, meaning there is more data in these tails than in the tails of a normal distribution. The positive skewness value indicates a right-tailed distribution (again, a value of 0.0 would indicate a normal distribution). The kernel density esimation, which can be thought of as a locally smoothed version of the histogram, is overlaid to help visualize the shape. Both components, the lepotkurtotic distribution and right-tailed distribution, can be seen in the visualization.  
  
The histogram to the right shows Serum Cholestorl with a Box-Cox transformation performed. Notice the much lower kurtosis value, although still postive and representing a lepotkurtotic distribution. The skewness value is nearly zero, representing much more normally-distributed data. Comparing the two histograms, it is evident the Box-Cox transformation was helpful in making the data into more of a normal distribution. This change makes it more useful for modeling purposes so the Box-Cox'd version of Serum Cholestorl will be used from here on out.

Including the details above, this step also involved:
* Statistical Analysis
   * Normality Tests
   * Chi-Square Tests
   * Contingency Tables
   * Odds Ratios
   * Descriptive Statistics
* Feature Engineering (do visualization will of this done?)
* Additional Data Visualizations
* Additonal Data Transformations [<sub><sup>View code</sup></sub>](#c)

## Model Building [<sub><sup>View code</sup></sub>](#d)
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

# Set aesthetic parameters of seaborn plots
sns.set()

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
Remove unnecessary columns  
```python
# List of columns to drop
cols_to_drop =['ccf', 'pncaden', 'smoke', 'cigs', 'years', 'dm', 'famhist', 'dig', 'ca', 'restckm', 'exerckm',
               'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'lmt',
               'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1',
               'cathef', 'junk', 'name', 'thaltime', 'xhypo', 'slope', 'dummy', 'lvx1', 'lvx2']

# Drop columns from above list
hungarian = hungarian.drop(columns=cols_to_drop)  
```

Convert column types  
```python
# Convert all columns to numeric
hungarian = hungarian.apply(pd.to_numeric)
```

Correct data discrepancies  
```python
### Fix possible patient id issues
# Find ids that are not unique to patients
print(hungarian.id.value_counts()[hungarian.id.value_counts()!=1])

# Fix id 1132 (two different patients are both assigned to this id) - give second patient next id number (id max + 1)
hungarian.loc[hungarian.loc[hungarian.id==1132].index[-1], 'id'] = hungarian.id.max() + 1
```

Remove patients with a large percentage of missing values  
```python
# Drop patients with "significant" number of missing values (use 10%, can adjust accordingly)
# Determine missing value percentage per patient (-9 is the missing attribute value)
missing_value_perc_per_patient = (hungarian == -9).sum(axis=1)[(hungarian == -9).sum(axis=1) > 0]\
                                     .sort_values(ascending=False)/len([x for x in hungarian.columns if x != 'id'])

# Remove patients with > 10% missing values
hungarian = hungarian.drop(missing_value_perc_per_patient[missing_value_perc_per_patient>0.10].index.values)
```
Impute missing values  
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
Set target variable to binary range
```python
# Set y variable to 0-1 range (as previous studies have done)
hungarian.loc[hungarian.num > 0, "num"] = 1
```

# c
```python
# Determine 'strong' alpha value based on sample size
sample_size_one, strong_alpha_value_one = 100, 0.001
sample_size_two, strong_alpha_value_two = 1000, 0.0003
slope = (strong_alpha_value_two - strong_alpha_value_one)/(sample_size_two - sample_size_one)
strong_alpha_value = slope * (hungarian.shape[0] - sample_size_one) + strong_alpha_value_one
print(f"The alpha value for use in hypothesis tests is {strong_alpha_value}.")

# List of continuous variables
continuous_variables = ['age', 'trestbps', 'chol', 'thaldur', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd',
                        'trestbpd', 'oldpeak', 'rldv5', 'rldv5e']

# List of categorical variables
categorical_variables = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop', 'nitr',
                         'pro', 'diuretic', 'proto', 'exang', 'lvx3', 'lvx4', 'lvf']

# Target variable
target_variable = 'num'

### Feature engineering ###

# Create column of time between ekg and cardiac cath
# Create column of ekg dates
ekg_date = []
for year, month, day in zip(hungarian.ekgyr, hungarian.ekgmo, hungarian.ekgday):
    x = str(year) + '-' + str(month) + '-' + str(day)
    ekg_date.append(dt.datetime.strptime(x, '%y-%m-%d').strftime('%Y-%m-%d'))
# Append list to datetime to create column
hungarian['ekg_date'] = ekg_date

# Correct 2-30-86 issue (1986 was not a leap year)
hungarian.loc[(hungarian.cyr==86) & (hungarian.cmo==2) & (hungarian.cday==30), ('cmo', 'cday')] = (3,1)

cardiac_cath_date = []
for year, month, day in zip(hungarian.cyr, hungarian.cmo, hungarian.cday):
    x = str(year) + '-' + str(month) + '-' + str(day)
    print(x)
    cardiac_cath_date.append(dt.datetime.strptime(x, '%y-%m-%d').strftime('%Y-%m-%d'))
# Append list to datetime to create column
hungarian['cardiac_cath_date'] = cardiac_cath_date

# Days between cardiac cath and ekg
hungarian['days_between_c_ekg'] = (pd.to_datetime(hungarian.cardiac_cath_date) - pd.to_datetime(hungarian.ekg_date)).dt.days

# Append days between cardiac cath and ekg to continuous variable list
continuous_variables.append('days_between_c_ekg')

# Create PCA variable from rldv5 and rldv5e
hungarian['rldv5_rldv5e_pca'] = PCA(n_components=1).fit_transform(hungarian[['rldv5', 'rldv5e']])

# Append new PCA'd variable to continuous variable list
continuous_variables.append('rldv5_rldv5e_pca')

# Dicitionary with continuous variable as key and spelled out version of variablea as value
continuous_variables_spelled_out_dict = {'age': 'Age', 'trestbps': 'Resting Blood Pressure (On Admission)',
                                         'chol': 'Serum Cholestoral', 'thaldur': 'Duration of Exercise Test (Minutes)',
                                         'met': 'METs Achieved', 'thalach': 'Maximum Heart Rate Achieved',
                                         'thalrest': 'Resting Heart Rate',
                                         'tpeakbps': 'Peak Exercise Blood Pressure (Systolic)',
                                         'tpeakbpd': 'Peak Exercise Blood Pressure (Diastolic)',
                                         'trestbpd': 'Resting Blood Pressure',
                                         'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
                                         'rldv5': 'Height at Rest',
                                         'rldv5e': 'Height at Peak Exercise',
                                         'days_between_c_ekg': 'Days Between Cardiac Catheterization and Electrocardiogram',
                                         'rldv5_rldv5e_pca': "PCA variable for 'Height at Rest' and 'Height at Peak Exercise'"}
```

Heatmap of Continous Predictor Variables
```python
# Heatmap of correlations
# Only return bottom portion of heatmap as top is duplicate and diagonal is redundant
continuous_variable_correlations = hungarian[continuous_variables].corr()
# Array of zeros with same shape as continuous_variable_correlations
mask = np.zeros_like(continuous_variable_correlations)
# Mark upper half and diagonal of mask as True
mask[np.triu_indices_from(mask)] = True
# Correlation heatmap
f, ax = plt.subplots(figsize=(9, 6))
f.subplots_adjust(left=0.32, right=0.89, top=0.95, bottom=0.32)
ax = sns.heatmap(hungarian[continuous_variables].corr(), cmap='PiYG', mask=mask, linewidths=.5, linecolor="white", cbar=True)
ax.set_xticklabels(labels=continuous_variables_spelled_out_dict.values(),fontdict ={'fontweight': 'bold', 'fontsize':10},
                   rotation=45, ha="right",
                   rotation_mode="anchor")
ax.set_yticklabels(labels=continuous_variables_spelled_out_dict.values(),fontdict ={'fontweight': 'bold', 'fontsize':10})
ax.set_title("Heatmap of Continuous Predictor Features", fontdict ={'fontweight': 'bold', 'fontsize': 22})
```
DataFrame of continuous variable correlations greater than 0.6 and less than -0.6 (more numerical alternative to above heatmap)
```python
print(hungarian[continuous_variables].corr()[(hungarian[continuous_variables].corr()>0.6) | (hungarian[continuous_variables].corr()<-0.6)])
```
Histograms of Continuous Features by Target
```python
fig, axes = plt.subplots(nrows=5, ncols=3)
fig.subplots_adjust(left=0.17, right=0.83, top=0.90, bottom=0.10, hspace=0.7, wspace = 0.25)
fig.suptitle('Distributions of Continuous Features by Target', fontweight='bold', fontsize= 22)
for ax, continuous in zip(axes.flatten(), continuous_variables):
    for num_value in hungarian.num.unique():
        ax.hist(hungarian.loc[hungarian.num == num_value, continuous], alpha=0.7, label=num_value)
        ax.set_title(continuous_variables_spelled_out_dict[continuous], fontdict ={'fontweight': 'bold', 'fontsize': 10})
handles, legends = ax.get_legend_handles_labels()
legends_spelled_out_dict = {0: "No Presence of Heart Disease", 1: "Presence of Heart Disease"}
fig.legend(handles, legends_spelled_out_dict.values(), loc='upper left', bbox_to_anchor=(0.68, 0.99), prop={'weight':'bold'})
```

Normality Tests
```python
# Check normality of continuous variables
for continuous in continuous_variables:
    print(continuous)
    print(f"Kurtosis value: {stats.kurtosis(a=hungarian[continuous], fisher=True)}")
    print(f"Sknewness value: {stats.skew(a=hungarian[continuous])}")
    print(f"P-value from normal test: {stats.normaltest(a=hungarian[continuous])[1]}")
    if stats.normaltest(a=hungarian[continuous])[1] < strong_alpha_value:
        print("Reject null hypothesis the samples comes from a normal distribution.")
        print("-------------------------------------------------------------------")
        try:
            print(f"Kurtosis value: {stats.kurtosis(a=stats.boxcox(x=hungarian[continuous])[0], fisher=True)}")
            print(f"Sknewness value: {stats.skew(a=stats.boxcox(x=hungarian[continuous])[0])}")
            print(f"P-value from normal test: {stats.normaltest(a=stats.boxcox(x=hungarian[continuous])[0])[1]}")
        except ValueError as a:
            if str(a) == "Data must be positive.":
                print(f"{continuous} contains zero or negative values.")
    else:
        print("Do not reject the null hypothesis")
    print('\n')
```

Data Transformations (Box-Cox Transformation)
```python
# Boxcox necessary variables that reject the null hypothesis from normaltest in scipy.stats
hungarian['trestbps_boxcox'] = stats.boxcox(x=hungarian.trestbps)[0]
hungarian['chol_boxcox'] = stats.boxcox(x=hungarian.chol)[0]
hungarian['thalrest_boxcox'] = stats.boxcox(x=hungarian.thalrest)[0]

# Add boxcox'd variables to continuous_variables_spelled_out_dict
for boxcox_var in filter(lambda x: '_boxcox' in x, hungarian.columns):
    continuous_variables_spelled_out_dict[boxcox_var] = continuous_variables_spelled_out_dict[
                                                            boxcox_var.split("_")[0]] + " Box-Cox"
```

Data Transformation of Serum Cholestrol - Distributions with KDE Overlaid
```python
# Compare original distribution with boxcox'd distribution for chol
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
fig.suptitle('Distributions with Kernel Density Estimation (KDE) Overlaid ', fontweight='bold', fontsize= 22)
for ax, variable in zip(axes.flatten(), ['chol', 'chol_boxcox']):
    print(ax, variable)
    ax.hist(hungarian[variable])
    ax2 = hungarian[variable].plot.kde(ax=ax, secondary_y=True)
    ax2.grid(False)
    ax2.set_yticks([])
    ax2.set_title(continuous_variables_spelled_out_dict[variable], fontdict={'fontweight': 'bold', 'fontsize': 24})
    ax.text(0.78, 0.75, f"Kurtosis value: {'{:.3}'.format(stats.kurtosis(a=hungarian[variable], fisher=True))}\n"
                      f"Sknewness value: {'{:.3}'.format(stats.skew(a=hungarian[variable]))}",
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='none', edgecolor='black', pad=10.0, linewidth=3), weight='bold', fontsize=14)
    ax.set_ylabel('Density', fontdict={'fontweight': 'bold', 'fontsize': 18})
# Expand figure to desired size first before running below code (this makes xtick labels appear and then can therefore be bolded)
for i in range(len(axes)):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), fontweight='bold')
```

Histograms of Continuous Features and their Box-Cox'd Versions by Target
```python
# Plot original and boxcox'd distributions to each other and against num
# Create list of boxcox'd variables and their originals
variables_for_inspection = list(itertools.chain.from_iterable([[x, x.split("_")[0]] for x in list(hungarian) if 'boxcox' in x]))
# Sort list in ascending order
variables_for_inspection.sort(reverse=False)
fig, axes = plt.subplots(nrows=len([x for x in variables_for_inspection if 'boxcox' in x]), ncols=2, figsize=(28,8))
fig.subplots_adjust(hspace=0.5)
fig.suptitle("Distributions of Continuous Features and their Box-Cox'd Versions", fontweight='bold', fontsize=20)
for ax, variable in zip(axes.flatten(), variables_for_inspection):
    for num_value in hungarian.num.unique():
        ax.hist(hungarian.loc[hungarian.num == num_value, variable], alpha=0.7, label=num_value)
        ax.set_title(continuous_variables_spelled_out_dict[variable], fontdict={'fontweight': 'bold', 'fontsize': 24})
handles, legends = ax.get_legend_handles_labels()
legends_spelled_out_dict = {0: "No Presence of Heart Disease", 1: "Presence of Heart Disease"}
fig.legend(handles, legends_spelled_out_dict.values(), loc='upper left', bbox_to_anchor=(0.77, 1.0),
           prop={'weight': 'bold', 'size': 14})
```

Chi-Square Tests
```python

# Pearson chi-square tests
chi_square_analysis_list = []
for categorical in categorical_variables:
    chi, p, dof, expected = stats.chi2_contingency(pd.crosstab(index=hungarian[categorical], columns=hungarian[target_variable]))
    print(f"The chi-square value for {categorical} and {target_variable} is {chi}, and the p-value is" f" {p}, respectfully.")
    chi_square_analysis_list.append([categorical, target_variable, chi, p])

# Create DataFrame from lists of lists
chi_square_analysis_df = pd.DataFrame(chi_square_analysis_list, columns=['variable', 'target', 'chi',
                                                            'p_value']).sort_values(by='p_value', ascending=True)
# Determine categorical variables that reject null
chi_square_analysis_df.loc[chi_square_analysis_df.p_value <= strong_alpha_value]
```
Contingency Tables, Odds Ratios, Descriptive Statistics, Chi-Square Tests
```python
# Crosstab of age and num
pd.crosstab(index=hungarian.age,columns=hungarian.num, normalize=True)

# Distribution plot of age of all patients
plt.figure()
sns.distplot(hungarian['age'], kde=True, fit=stats.norm, rug=False,
             kde_kws={"label": "Kernel Density Esimation (KDE)"},
             fit_kws={"label": "Normal Distribution"}).set_title("Age Distribution of Patients")
plt.legend(loc='best')
plt.show()

# Statistical understanding of age of all patients
print(f"Mean +/- std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'],2)}. This means 68% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - hungarian['age'].describe()['std'],2)} and"
      f" {round(hungarian['age'].describe()['mean'] + hungarian['age'].describe()['std'],2)}.")
standard_devations = 2
print(f"Mean +/- {standard_devations} std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'] * standard_devations,2)}. This means 95% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - (standard_devations * hungarian['age'].describe()['std']),2)} and"
      f" {round(hungarian['age'].describe()['mean'] + (standard_devations * hungarian['age'].describe()['std']),2)}.")
standard_devations = 3
print(f"Mean +/- {standard_devations} std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'] * standard_devations,2)}. This means 99.7% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - (standard_devations * hungarian['age'].describe()['std']),2)} and"
      f" {round(hungarian['age'].describe()['mean'] + (standard_devations * hungarian['age'].describe()['std']),2)}.")
print(f"Mode of {hungarian['age'].name}: {hungarian['age'].mode()[0]}\nMedian of {hungarian['age'].name}: {hungarian['age'].median()}")

# Distribution plot of age of patients broken down by sex - female
plt.figure()
sns.distplot(hungarian.loc[hungarian['sex']==0, 'age'], kde=True, fit=stats.norm,
             kde_kws={"label": "Kernel Density Esimation (KDE)"},
             fit_kws={"label": "Normal Distribution"}).set_title('Female Age Distribution')
plt.legend(loc='best')
plt.show()

# Women age information
print(f"Mean +/- std of {hungarian.loc[hungarian['sex']==0, 'age'].name} for women: "
      f"{round(hungarian.loc[hungarian['sex']==0, 'age'].describe()['mean'],2)} +/ "
      f"{round(hungarian.loc[hungarian['sex']==0, 'age'].describe()['std'],2)}")

print(f"Mode of {hungarian.loc[hungarian['sex']==0, 'age'].name} for women: "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].mode()[0]}\nMedian of "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].name} for women: + "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].median()}")
print('\n')

# Distribution plot of age of patients broken down by sex - male
plt.figure()
sns.distplot(hungarian.loc[hungarian['sex']==1, 'age'],kde=True, fit=stats.norm,
             kde_kws={"label": "Kernel Density Esimation (KDE)"},
             fit_kws={"label": "Normal Distribution"}).set_title('Male Age Distribution')
plt.legend(loc='best')
plt.show()

# Men age information
print(f"Mean +/- std of {hungarian.loc[hungarian['sex']==1, 'age'].name} for men: "
      f"{round(hungarian.loc[hungarian['sex']==1, 'age'].describe()['mean'],2)} +/ "
      f"{round(hungarian.loc[hungarian['sex']==1, 'age'].describe()['std'],2)}")

print(f"Mode of {hungarian.loc[hungarian['sex']==1, 'age'].name} for men: "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].mode()[0]}\nMedian of "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].name} for men: + "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].median()}")

## Sex
# Get counts of sex
print(hungarian.sex.value_counts())
print(f'The hungarian dataset consists of {hungarian.sex.value_counts()[0]} females and'
      f' {hungarian.sex.value_counts()[1]} males.')

# Bar graph of sex by num
plt.figure()
sex_dict = {0: "female", 1: "male"}
sns.countplot(x="sex", hue="num", data=hungarian).set(title='Heart Disease Indicator by Sex', xticklabels=sex_dict.values())
plt.show()

# Crosstab of sex by num
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num))

# Crosstab of sex by num - all values normalized
# Of all patients in dataset, 32% were males that had heart disease. 4% were females that had heart disease.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='all'))

# Crosstab of sex by num - rows normalized
# 15% of females had heart disease. 44% of males had heart disease.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='index'))

# Crosstab of sex by num - columns normalized
# 89% of the patients with heart disease were males. 11% were females.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='columns'))

# Contingency table of sex by num
contingency = pd.crosstab(index=hungarian.sex, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)

if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same for male and female patients.")
else:
    print(f"Fail to reject the null of no association between sex and diagnosis of heart disease. The probability of a "
          f"heart disease diagnosis is the same regardless of a patient's sex.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means males are {round(table.oddsratio,2)} times more likely to be "
      f"diagnosed with heart disease than females.")

## Painloc
# Bar graph of painloc by num
plt.figure()
painloc_dict = {0: "otherwise", 1: "substernal"}
sns.countplot(x="painloc", hue="num", data=hungarian).set(title='Heart Disease Indicator by Pain Location', xticklabels=painloc_dict.values())
plt.show()

# Contingency table of painloc by num
contingency = pd.crosstab(index=hungarian.painloc, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same based on chest pain location.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of chest pain location.")


# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}.")

## Painexer
# Bar graph of painexer by num
plt.figure()
painexer_dict = {0: "otherwise", 1: "provoked by exertion"}
sns.countplot(x="painexer", hue="num",
              data=hungarian).set(title='Heart Disease Indicator by Pain Exertion', xticklabels=painexer_dict.values())
plt.show()

# Contingency table of painexer by num
contingency = pd.crosstab(index=hungarian.painexer, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully. The expected values are\n{expected}.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same based on how chest pain is provoked.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of how chest pain is provoked.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means patients with their chest pain provoked by exertion are "
      f"{round(table.oddsratio,2)} times more likely to have a diagnosis of heart disease than those patients with "
      f"their chest pain provoked otherwise.")

## Relrest
# Bar graph of relrest by num
plt.figure()
relrest_dict = {0: "otherwise", 1: "relieved after rest"}
sns.countplot(x="relrest", hue="num", data=hungarian).set(title='Heart Disease Indicator by Pain Relief', xticklabels=relrest_dict.values())
plt.show()

# Contingency table of relrest by num
contingency = pd.crosstab(index=hungarian.relrest, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully. The expected values are\n{expected}.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same for pain relieved after rest and "
          f"otherwise.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of when the pain is relieved.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means patients with their chest pain relieved after rest are "
      f"{round(table.oddsratio,2)} times more likely to have a diagnosis of heart disease than those patients with "
      f"their chest pain relieved otherwise.")

## Cp
# Bar graph of cp by num
plt.figure()
cp_dict = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}
sns.countplot(x="cp", hue="num", data=hungarian).set(title='Heart Disease Indicator by Chest Pain Type', xticklabels=cp_dict.values())
plt.show()

# Contingency table of cp by cum
contingency = pd.crosstab(index=hungarian.cp, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same depending on chest pain type.")
else:
    print(
        f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
        f"The probability of a heart disease diagnosis is the same regardless of chest pain type.")
```
Feature Engineering
```python
hungarian["thalach_div_by_thalrest"] = hungarian["thalach"]/hungarian["thalrest"]
hungarian["tpeakbps_div_by_tpeakbpd"] = hungarian["tpeakbps"]/hungarian["tpeakbpd"]
hungarian["thaldur_div_by_met"] = hungarian["thaldur"]/hungarian["met"]
hungarian["chol_div_by_age"] = hungarian["chol"]/hungarian["age"]
hungarian["chol_div_by_met"] = hungarian["chol"]/hungarian["met"]
hungarian["chol_div_by_thalach"] = hungarian["chol"]/hungarian["thalach"]
hungarian["chol_div_by_thalrest"] = hungarian["chol"]/hungarian["thalrest"]
hungarian["thalrest_div_by_rldv5"] = hungarian["thalrest"]/hungarian["rldv5"]
hungarian["thalach_div_by_rldv5e"] = hungarian["thalrest"]/hungarian["rldv5e"]

hungarian["trestbps_boxcox_div_by_tpeakbpd"] = hungarian["trestbps_boxcox"]/hungarian["tpeakbpd"]

hungarian["chol_boxcox_div_by_age"] = hungarian["chol_boxcox"]/hungarian["age"]
hungarian["chol_boxcox_div_by_met"] = hungarian["chol_boxcox"]/hungarian["met"]
hungarian["chol_boxcox_div_by_thalach"] = hungarian["chol_boxcox"]/hungarian["thalach"]
hungarian["chol_boxcox_div_by_thalrest"] = hungarian["chol_boxcox"]/hungarian["thalrest"]

hungarian["thalach_div_by_thalrest_boxcox"] = hungarian["thalach"]/hungarian["thalrest_boxcox"]
hungarian["chol_div_by_thalrest_boxcox"] = hungarian["chol"]/hungarian["thalrest_boxcox"]
hungarian["thalrest_boxcox_div_by_rldv5"] = hungarian["thalrest_boxcox"]/hungarian["rldv5"]
# Bin age
hungarian['agebinned'] = pd.cut(x=hungarian.age, bins=5, labels = ['0', '1', '2', '3', '4'])
```

DataFrame of continuous variable correlations greater than 0.6 and less than 1.0 and less than -0.6 and greater than -1.0 with entirely null columns dropped - all feature engineered variables included
```python
# Add boxcox'd variables to continuous_variables list
continuous_variables.extend([x for x in list(hungarian) if 'boxcox' in x])
# Add iteraction variables to continuous_variables list
continuous_variables.extend([x for x in list(hungarian) if 'div_by' in x])

# Correlations > 0.6 and < 1.0 and <-0.6 and >-1.0, drop all null columns
hungarian[continuous_variables].corr()[((hungarian[continuous_variables].corr() > 0.6) & 
                                              (hungarian[continuous_variables].corr() < 1.0)) | 
                                             ((hungarian[continuous_variables].corr()<-0.6) & 
                                              (hungarian[continuous_variables].corr()>-1.0))].dropna(axis=1, how='all')
```

# d

DataFrame to append model results
```python
# Create empty DataFrame to append all model results to
all_model_results = pd.DataFrame()
# Create DataFrame to append top model results to
top_model_results = pd.DataFrame(columns=['model_type', 'solver', 'best_model_params_grid_search', 'best_score_grid_search',
                                          'true_negatives', 'false_positives', 'false_negatives', 'true_positives',
                                          'recall', 'precision', 'f1_score', 'variables_not_used', 'variables_used',
                                          'model_params_grid_search'])
```

Copy of patient DataFrame for regression modeling
```python
model = hungarian.copy()
```

Create unique set of variables
```python
# Drop columns
variables_to_drop_for_modeling_one = ['id', 'ekgyr', 'ekgmo', 'ekgday', 'cyr', 'cmo', 'cday', 'lvx3', 'lvx4', 'lvf',
                                      'proto', 'ekg_date', 'cardiac_cath_date', 'rldv5_rldv5e_pca',
                                      'days_between_c_ekg', 'trestbps_boxcox', 'chol_boxcox', 'thalrest_boxcox',
                                      'thalach_div_by_thalrest', 'tpeakbps_div_by_tpeakbpd', 'thaldur_div_by_met',
                                      'chol_div_by_met', 'chol_div_by_thalach', 'chol_div_by_thalrest',
                                      'chol_div_by_age', 'thalrest_div_by_rldv5', 'thalach_div_by_rldv5e', 'agebinned',
                                      'trestbps_boxcox_div_by_tpeakbpd', 'chol_boxcox_div_by_age',
                                      'chol_boxcox_div_by_met', 'chol_boxcox_div_by_thalach',
                                      'chol_boxcox_div_by_thalrest', 'thalach_div_by_thalrest_boxcox',
                                      'chol_div_by_thalrest_boxcox', 'thalrest_boxcox_div_by_rldv5']
# Define categorical variables
categorical_variables_for_modeling_one = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg',
                                          'prop', 'nitr', 'pro', 'diuretic', 'exang']


# Drop columns
variables_to_drop_for_modeling_two = ['id', 'chol', 'thalrest', 'trestbps', 'ekgyr', 'ekgmo', 'ekgday', 'cyr', 'cmo',
                                      'cday', 'ekg_date', 'cardiac_cath_date', 'rldv5', 'lvx3', 'lvx4', 'lvf', 'pro',
                                      'proto', 'rldv5_rldv5e_pca', 'thalach_div_by_thalrest',
                                      'tpeakbps_div_by_tpeakbpd', 'thaldur_div_by_met', 'chol_div_by_met',
                                      'chol_div_by_thalach', 'chol_div_by_thalrest', 'chol_div_by_age',
                                      'thalrest_div_by_rldv5', 'thalach_div_by_rldv5e', 'agebinned',
                                      'trestbps_boxcox_div_by_tpeakbpd', 'chol_boxcox_div_by_age',
                                      'chol_boxcox_div_by_met', 'chol_boxcox_div_by_thalach',
                                      'chol_boxcox_div_by_thalrest', 'thalach_div_by_thalrest_boxcox',
                                      'chol_div_by_thalrest_boxcox', 'thalrest_boxcox_div_by_rldv5']
# Define categorical variables
categorical_variables_for_modeling_two = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg',
                                          'prop', 'nitr', 'diuretic', 'exang']

# Drop columns
variables_to_drop_for_modeling_three = ['id', 'chol', 'thalrest', 'trestbps', 'ekgyr', 'ekgmo', 'ekgday', 'cyr', 'cmo',
                                      'cday', 'ekg_date', 'cardiac_cath_date', 'lvx3', 'lvx4', 'lvf', 'proto',
                                      'rldv5_rldv5e_pca', 'thalach_div_by_thalrest',
                                      'tpeakbps_div_by_tpeakbpd', 'thaldur_div_by_met', 'chol_div_by_met',
                                      'chol_div_by_thalach', 'chol_div_by_thalrest', 'chol_div_by_age',
                                      'thalrest_div_by_rldv5', 'thalach_div_by_rldv5e', 'agebinned',
                                      'trestbps_boxcox_div_by_tpeakbpd', 'chol_boxcox_div_by_age',
                                      'chol_boxcox_div_by_met', 'chol_boxcox_div_by_thalach',
                                      'chol_boxcox_div_by_thalrest', 'thalach_div_by_thalrest_boxcox',
                                      'chol_div_by_thalrest_boxcox', 'thalrest_boxcox_div_by_rldv5']

# Define categorical variables
categorical_variables_for_modeling_three = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop',
                                          'nitr', 'diuretic', 'exang', 'pro']

# Drop columns
variables_to_drop_for_modeling_four = ['id', 'ekgyr', 'ekgmo', 'ekgday', 'cyr', 'cmo', 'cday', 'lvx3', 'lvx4', 'lvf',
                                      'proto', 'ekg_date', 'cardiac_cath_date', 'rldv5', 'rldv5e', 'trestbps_boxcox',
                                       'chol_boxcox', 'thalrest_boxcox', 'agebinned',
                                       'trestbps_boxcox_div_by_tpeakbpd', 'chol_boxcox_div_by_age',
                                       'chol_boxcox_div_by_met', 'chol_boxcox_div_by_thalach',
                                       'chol_boxcox_div_by_thalrest', 'thalach_div_by_thalrest_boxcox',
                                       'chol_div_by_thalrest_boxcox', 'thalrest_boxcox_div_by_rldv5']
# Define categorical variables
categorical_variables_for_modeling_four = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop',
                                          'nitr', 'diuretic', 'exang', 'pro']

# Drop columns
variables_to_drop_for_modeling_five = ['id', 'age', 'ekgyr', 'ekgmo', 'ekgday', 'cyr', 'cmo', 'cday', 'lvx3', 'lvx4', 'lvf',
                                      'proto', 'ekg_date', 'cardiac_cath_date','rldv5_rldv5e_pca', 'trestbps_boxcox',
                                       'chol_boxcox', 'thalrest_boxcox', 'trestbps_boxcox_div_by_tpeakbpd',
                                       'chol_boxcox_div_by_age', 'chol_boxcox_div_by_met', 'chol_boxcox_div_by_thalach',
                                       'chol_boxcox_div_by_thalrest', 'thalach_div_by_thalrest_boxcox',
                                       'chol_div_by_thalrest_boxcox', 'thalrest_boxcox_div_by_rldv5']

# Define categorical variables
categorical_variables_for_modeling_five = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop',
                                          'nitr', 'diuretic', 'exang', 'pro', 'agebinned']

# Drop columns
variables_to_drop_for_modeling_six = ['id', 'age', 'chol', 'thalrest', 'trestbps', 'ekgyr', 'ekgmo', 'ekgday', 'cyr',
                                      'cmo', 'cday', 'lvx3', 'lvx4', 'lvf',
                                      'proto', 'ekg_date', 'cardiac_cath_date','rldv5_rldv5e_pca',
                                      'thalach_div_by_thalrest', 'tpeakbps_div_by_tpeakbpd', 'thaldur_div_by_met',
                                      'chol_div_by_met', 'chol_div_by_thalach', 'chol_div_by_thalrest',
                                      'chol_div_by_age', 'thalrest_div_by_rldv5', 'thalach_div_by_rldv5e']

# Define categorical variables
categorical_variables_for_modeling_six = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop',
                                          'nitr', 'diuretic', 'exang', 'pro', 'agebinned']

# Drop columns
variables_to_drop_for_modeling_seven = ['id', 'chol', 'thalrest', 'trestbps', 'ekgyr', 'ekgmo', 'ekgday', 'cyr',
                                      'cmo', 'cday', 'lvx3', 'lvx4', 'lvf',
                                      'proto', 'ekg_date', 'cardiac_cath_date','rldv5_rldv5e_pca',
                                      'thalach_div_by_thalrest', 'tpeakbps_div_by_tpeakbpd', 'thaldur_div_by_met',
                                      'chol_div_by_met', 'chol_div_by_thalach', 'chol_div_by_thalrest',
                                      'chol_div_by_age', 'thalrest_div_by_rldv5', 'thalach_div_by_rldv5e', 'agebinned']

# Define categorical variables
categorical_variables_for_modeling_seven = ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop',
                                          'nitr', 'diuretic', 'exang', 'pro']

# Make list of lists for variables to drop
variables_to_drop_list = [variables_to_drop_for_modeling_one, variables_to_drop_for_modeling_two,
                          variables_to_drop_for_modeling_three, variables_to_drop_for_modeling_four,
                          variables_to_drop_for_modeling_five,variables_to_drop_for_modeling_six,
                          variables_to_drop_for_modeling_seven]

# Make list of lists for categorical variables to model
categorical_variables_for_modeling_list = [categorical_variables_for_modeling_one,
                                           categorical_variables_for_modeling_two,
                                           categorical_variables_for_modeling_three,
                                           categorical_variables_for_modeling_four,
                                           categorical_variables_for_modeling_five,
                                           categorical_variables_for_modeling_six,
                                           categorical_variables_for_modeling_seven]
```
Logistic Regression
```python
# Unique variable combination runs
for index, (vars_to_drop, cat_vars_to_model) in enumerate(zip(variables_to_drop_list,
                                                              categorical_variables_for_modeling_list), start=1):
    print(f"Model run: {index}")
    # Create copy of hungarian for regression modeling
    model = hungarian.copy()
    # Drop variables
    model = model.drop(columns=vars_to_drop)
    # Dummy variable categorical variables
    model = pd.get_dummies(data=model, columns=cat_vars_to_model, drop_first=True)
    # Create target variable
    y = model['num']
    # Create feature variables
    x = model.drop(columns='num')

    # Obtain recursive feature elimination values for all solvers and get average
    # (not sure what to do about ConvergenceWarning - get warning but also get result for each solver)
    rfe_logit = pd.DataFrame(data=list(x), columns=['variable'])
    for solve in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
        rfe_logit = rfe_logit.merge(pd.DataFrame(data=[list(x), RFE(LogisticRegression(solver=solve, max_iter=100),
                    n_features_to_select=1).fit(x, y).ranking_.tolist()]).T.rename(columns={0: 'variable', 1:
                    'rfe_ranking_' + solve}), on='variable')
    # Get average ranking for each variable
    rfe_logit['rfe_ranking_avg'] = rfe_logit[['rfe_ranking_liblinear', 'rfe_ranking_newton-cg', 'rfe_ranking_lbfgs',
                                              'rfe_ranking_sag', 'rfe_ranking_saga']].mean(axis=1)
    # Sort DataFrame
    rfe_logit = rfe_logit.sort_values(by='rfe_ranking_avg', ascending=True).reset_index(drop=True)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=43)
    # Run models - start at top and add variables with each iteration
    # Test 'weaker' alpha value
    strong_alpha_value = 0.04
    model_search_logit = []
    logit_variable_list = []
    insignificant_variables_list = []
    for i in range(len(rfe_logit)):
        if rfe_logit['variable'][i] not in logit_variable_list and rfe_logit['variable'][i] not in insignificant_variables_list:
            logit_variable_list.extend([rfe_logit['variable'][i]])
            # logit_variable_list = list(set(logit_variable_list).difference(set(insignificant_variables_list)))
            logit_variable_list = [x for x in logit_variable_list if x not in insignificant_variables_list]
            print(logit_variable_list)
            # Add related one-hot encoded variables if variable is categorical
            if logit_variable_list[-1].split('_')[-1] in sorted([x for x in list(set([x.split('_')[-1] for x in list(x)])) if len(x) == 1]):
                logit_variable_list.extend([var for var in list(x) if logit_variable_list[-1].split('_')[0] in var and var != logit_variable_list[-1]])
                print(logit_variable_list)
            # Build logistic regression
            sm_logistic = sm.Logit(y_train, x_train[logit_variable_list]).fit()
            # All p-values are significant
            if all(p_values < strong_alpha_value for p_values in sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values):
                print("-*"*60)
                print((sm_logistic.summary2().tables[1]._getitem_column("P>|z|").index.tolist(),
                                         sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values.tolist()))
                print("-*"*60)
                print("-*"*60)
                model_search_logit.append([(sm_logistic.summary2().tables[0][0][6], sm_logistic.summary2().tables[0][1][6]),
                                        (sm_logistic.summary2().tables[0][2][0], sm_logistic.summary2().tables[0][3][0]),
                                        (sm_logistic.summary2().tables[0][2][1], sm_logistic.summary2().tables[0][3][1]),
                                        (sm_logistic.summary2().tables[0][2][2], sm_logistic.summary2().tables[0][3][2]),
                                        (sm_logistic.summary2().tables[0][2][3], sm_logistic.summary2().tables[0][3][3]),
                                        (sm_logistic.summary2().tables[0][2][4], sm_logistic.summary2().tables[0][3][4]),
                                        (sm_logistic.summary2().tables[0][2][5], sm_logistic.summary2().tables[0][3][5]),
                                        (sm_logistic.summary2().tables[1]._getitem_column("P>|z|").index.tolist(),
                                         sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values.tolist())])
            # P-value(s) of particular variable(s) is not significant
            elif any(p_values > strong_alpha_value for p_values in sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values):
                print('*'*60)
                print(logit_variable_list[-1])
                print('*'*60)
                if logit_variable_list[-1].split('_')[-1] in sorted([x for x in list(set([x.split('_')[-1] for x in list(x)])) if len(x) == 1]):
                    cat_var_level_check = sm_logistic.summary2().tables[1]._getitem_column("P>|z|")[sm_logistic.summary2().
                        tables[1]._getitem_column("P>|z|").index.isin([var for var in list(x) if
                                                                       logit_variable_list[-1].split('_')[0] in var])]
                    # If True, at least one level of the categorical variable is significant so keep all levels of variable
                    if any(p_values < strong_alpha_value for p_values in cat_var_level_check.values):
                        model_search_logit.append([(sm_logistic.summary2().tables[0][0][6], sm_logistic.summary2().tables[0][1][6]),
                                        (sm_logistic.summary2().tables[0][2][0], sm_logistic.summary2().tables[0][3][0]),
                                        (sm_logistic.summary2().tables[0][2][1], sm_logistic.summary2().tables[0][3][1]),
                                        (sm_logistic.summary2().tables[0][2][2], sm_logistic.summary2().tables[0][3][2]),
                                        (sm_logistic.summary2().tables[0][2][3], sm_logistic.summary2().tables[0][3][3]),
                                        (sm_logistic.summary2().tables[0][2][4], sm_logistic.summary2().tables[0][3][4]),
                                        (sm_logistic.summary2().tables[0][2][5], sm_logistic.summary2().tables[0][3][5]),
                                        (sm_logistic.summary2().tables[1]._getitem_column("P>|z|").index.tolist(),
                                         sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values.tolist())])
                    # Else False - remove all levels of categorical variable
                    else:
                        print("-"*60)
                        print(sm_logistic.summary2())
                        insignificant_variables_list.extend(cat_var_level_check.index)
                else:
                    print('='*60)
                    print(sm_logistic.summary2())
                    print(logit_variable_list[-1])
                    cont_var_check = sm_logistic.summary2().tables[1]._getitem_column("P>|z|")[sm_logistic.summary2().
                        tables[1]._getitem_column("P>|z|").index.isin([logit_variable_list[-1]])]
                    # Continuous variable is significant
                    if cont_var_check.values[0] < strong_alpha_value:
                        model_search_logit.append([(sm_logistic.summary2().tables[0][0][6], sm_logistic.summary2().tables[0][1][6]),
                                        (sm_logistic.summary2().tables[0][2][0], sm_logistic.summary2().tables[0][3][0]),
                                        (sm_logistic.summary2().tables[0][2][1], sm_logistic.summary2().tables[0][3][1]),
                                        (sm_logistic.summary2().tables[0][2][2], sm_logistic.summary2().tables[0][3][2]),
                                        (sm_logistic.summary2().tables[0][2][3], sm_logistic.summary2().tables[0][3][3]),
                                        (sm_logistic.summary2().tables[0][2][4], sm_logistic.summary2().tables[0][3][4]),
                                        (sm_logistic.summary2().tables[0][2][5], sm_logistic.summary2().tables[0][3][5]),
                                        (sm_logistic.summary2().tables[1]._getitem_column("P>|z|").index.tolist(),
                                         sm_logistic.summary2().tables[1]._getitem_column("P>|z|").values.tolist())])
                    else:
                        print('^'*60)
                        print(logit_variable_list[-1])
                        insignificant_variables_list.append(logit_variable_list[-1])
    # Create DataFrame of logisitic regression results
    model_search_logit = pd.DataFrame(model_search_logit, columns = ['converged', 'pseudo_r_squared', 'aic', 'bic',
                                                'log_likelihood', 'll_null', 'llr_p_value', 'columns_significance'])
    model_results_logit = []
    for solve in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
        for col in model_search_logit['columns_significance']:
            print(solve, col[0])
            try:
                logit_predict = cross_val_predict(LogisticRegression(solver=solve, max_iter=100), x[col[0]], y, cv=5)
                print(confusion_matrix(y_true=y, y_pred=logit_predict))
                conf_matr = confusion_matrix(y_true=y, y_pred=logit_predict)
                model_results_logit.append([solve, col[0], conf_matr[0][0], conf_matr[0][1], conf_matr[1][0], conf_matr[1][1]])
            except ConvergenceWarning:
                print("#"*60)
    # Create DataFrame of results
    model_results_logit = pd.DataFrame(model_results_logit, columns = ['solver', 'variables_used', 'true_negatives', 'false_positives',
                                                 'false_negatives', 'true_positives'])
    # Create recall, precision, and f1-score columns
    model_results_logit['recall'] = model_results_logit.true_positives/(model_results_logit.true_positives + model_results_logit.false_negatives)
    model_results_logit['precision'] = model_results_logit.true_positives/(model_results_logit.true_positives + model_results_logit.false_positives)
    model_results_logit['f1_score'] = 2 * (model_results_logit.precision * model_results_logit.recall) / (model_results_logit.precision + model_results_logit.recall)
    # Sort DataFrame
    model_results_logit = model_results_logit.sort_values(by=['f1_score'], ascending=False)
    print(model_results_logit)

    if len(model_results_logit.loc[model_results_logit.f1_score==model_results_logit.f1_score.max()]) > 1:
        top_model_result_logit = model_results_logit.loc[(model_results_logit.f1_score == model_results_logit.f1_score.max()) &
            (model_results_logit['variables_used'].apply(len) == min(map(lambda x: len(x[[1]][0]),
            model_results_logit.loc[model_results_logit.f1_score==model_results_logit.f1_score.max()].values)))].sample(n=1)
    else:
        top_model_result_logit = model_results_logit.loc[model_results_logit.f1_score == model_results_logit.f1_score.max()]
    top_model_results = top_model_results.append(other= top_model_result_logit, sort=False)
    print(f"Top logit model: \n {top_model_result_logit}")

    # Append top_model_result_logit results to all_model_results DataFrame
    logit_predict_proba = cross_val_predict(LogisticRegression(solver=top_model_result_logit["solver"].values[0],
                                max_iter=100), x[top_model_result_logit["variables_used"].values[0]], y, cv=5, method="predict_proba")
    all_model_results['logit_'+inflect.engine().number_to_words(index)+'_pred_zero'] = logit_predict_proba[:,0]
    all_model_results['logit_'+inflect.engine().number_to_words(index)+'_pred_one'] = logit_predict_proba[:,1]

# Fill in model_type columns
top_model_results['model_type'] = top_model_results['model_type'].fillna(value='logit')
```
Random Forest
```python
# Unique variable combination runs
for index, (vars_to_drop, cat_vars_to_model) in enumerate(zip(variables_to_drop_list,
                                                              categorical_variables_for_modeling_list), start=1):
    print(f"Model run: {index}")
    # Create copy of hungarian for non-regression modeling
    model = hungarian.copy()
    # Drop columns
    model = model.drop(columns=vars_to_drop)
    # Dummy variable categorical variables
    model = pd.get_dummies(data=model, columns=cat_vars_to_model, drop_first=False)
    # Create target variable
    y = model['num']
    # Create feature variables
    x = model.drop(columns='num')

    # Define parameters of Random Forest Classifier
    random_forest_model = RandomForestClassifier(random_state=1)
    # Define parameters for grid search
    param_grid = {'n_estimators': np.arange(10, 111, step=5), 'criterion': ['gini', 'entropy'],
                  'max_features': np.arange(2, 25, step=3)}
    cv = ShuffleSplit(n_splits=5, test_size=0.3)

    # Define grid search CV parameters
    grid_search = GridSearchCV(random_forest_model, param_grid, cv=cv) # , scoring='recall' # warm_start=True
    # Loop to iterate through least important variables according to random_forest_feature_importance and grid search
    x_all = list(x)
    model_search_rfc = []
    while True:
        print("--------------------------------")
        print(len(list(x)))
        print(print(param_grid['max_features']))
        print("--------------------------------")
        # try:
        grid_search.fit(x, y)
        print(f'Best parameters for current grid seach: {grid_search.best_params_}')
        print(f'Best score for current grid seach: {grid_search.best_score_}')
        # Define parameters of Random Forest Classifier from grid search
        random_forest_model = RandomForestClassifier(criterion=grid_search.best_params_['criterion'],
                                                     max_features=grid_search.best_params_['max_features'],
                                                     n_estimators=grid_search.best_params_['n_estimators'],
                                                     random_state=1)
        # Cross-validate and predict using Random Forest Classifer
        random_forest_predict = cross_val_predict(random_forest_model, x, y, cv=5)
        print(confusion_matrix(y_true=y, y_pred=random_forest_predict))
        conf_matr = confusion_matrix(y_true=y, y_pred=random_forest_predict)
        model_search_rfc.append([grid_search.best_params_, grid_search.best_score_, conf_matr[0][0], conf_matr[0][1],
                                 conf_matr[1][0], conf_matr[1][1], set(x_all).difference(x)])
        # Run random forest with parameters from grid search to obtain feature importances
        random_forest_feature_importance = pd.DataFrame(data=[list(x),
                    RandomForestClassifier(criterion=grid_search.best_params_['criterion'],
                     max_features=grid_search.best_params_['max_features'],
                     n_estimators=grid_search.best_params_['n_estimators'], random_state=1).fit(x,y).feature_importances_.tolist()]).T.rename(columns={0:'variable',
                     1:'importance'}).sort_values(by='importance', ascending=False)
        print(random_forest_feature_importance)
        if len(random_forest_feature_importance.loc[random_forest_feature_importance.importance<0.01]) > 0:
            for i in range(1, len(random_forest_feature_importance.loc[random_forest_feature_importance.importance<0.01])+1):
                print(f"'Worst' variable being examined: {random_forest_feature_importance.loc[random_forest_feature_importance.importance<0.01].variable.values[-i]}")
                bottom_variable = random_forest_feature_importance.loc[random_forest_feature_importance.importance<0.01].variable.values[-i]
                bottom_variable = bottom_variable.split('_')[0]
                bottom_variable = [col for col in list(x) if bottom_variable in col]
                compare_counter = 0
                for var in bottom_variable:
                    if var in random_forest_feature_importance.loc[random_forest_feature_importance.importance<0.01].variable.values:
                        compare_counter += 1
                if len(bottom_variable) == compare_counter:
                    print(f"Following variable(s) will be dropped from x {bottom_variable}")
                    x = x.drop(columns=bottom_variable)
                    break
                else:
                    print("Next 'worst' variable will be examined for dropping.")
                    continue
            else:
                break
    # Create DataFrame of random forest classifer results
    model_search_rfc = pd.DataFrame(model_search_rfc, columns=['best_model_params_grid_search', 'best_score_grid_search',
                                                 'true_negatives', 'false_positives',
                                                 'false_negatives', 'true_positives', 'variables_not_used'])
        # Create recall and precision columns
    model_search_rfc['recall'] = model_search_rfc.true_positives/(model_search_rfc.true_positives + model_search_rfc.false_negatives)
    model_search_rfc['precision'] = model_search_rfc.true_positives/(model_search_rfc.true_positives + model_search_rfc.false_positives)
    model_search_rfc['f1_score'] = 2 * (model_search_rfc.precision * model_search_rfc.recall) / (model_search_rfc.precision + model_search_rfc.recall)
    # Sort DataFrame
    model_search_rfc = model_search_rfc.sort_values(by=['f1_score'], ascending=False)
    print(model_search_rfc)

    if len(model_search_rfc.loc[model_search_rfc.f1_score==model_search_rfc.f1_score.max()]) > 1:
        top_model_result_rfc = model_search_rfc.loc[(model_search_rfc.f1_score==model_search_rfc.f1_score.max()) &
            (model_search_rfc['variables_not_used'].apply(len) == max(map(lambda x: len(x[list(model_search_rfc).index('variables_not_used')]),
             model_search_rfc.loc[model_search_rfc.f1_score==model_search_rfc.f1_score.max()].values)))]
        if len(top_model_result_rfc) > 1:
            print("Fix multiple best model problem for rfc")
            break
    else:
        top_model_result_rfc = model_search_rfc.loc[model_search_rfc.f1_score==model_search_rfc.f1_score.max()]
    top_model_results = top_model_results.append(other=top_model_result_rfc, sort=False)
    print(f"Top rfc model: \n {top_model_result_rfc}")

    # Append top_model_result_rfc results to all_model_results DataFrame
    # Re-create feature variables
    x = model.drop(columns='num')
    rfc_predict_proba = cross_val_predict(RandomForestClassifier(criterion=top_model_result_rfc["best_model_params_grid_search"].values[0]['criterion'],
                     max_features=top_model_result_rfc["best_model_params_grid_search"].values[0]['max_features'],
                     n_estimators=top_model_result_rfc["best_model_params_grid_search"].values[0]['n_estimators']),
                     x[[x for x in list(x) if x not in list(top_model_result_rfc['variables_not_used'].values[0])]], y,
                                          cv=5, method='predict_proba')
    all_model_results['rfc_'+inflect.engine().number_to_words(index)+'_pred_zero'] = rfc_predict_proba[:,0]
    all_model_results['rfc_'+inflect.engine().number_to_words(index)+'_pred_one'] = rfc_predict_proba[:,1]

# Fill in model_type columns
top_model_results['model_type'] = top_model_results['model_type'].fillna(value='rfc')
```

Support Vector Machine
```python
# Unique variable combination runs
for index, (vars_to_drop, cat_vars_to_model) in enumerate(zip(variables_to_drop_list,
                                                              categorical_variables_for_modeling_list), start=1):
    print(f"Model run: {index}")
    # Create copy of hungarian for non-regression modeling
    model = hungarian.copy()
    # Drop columns
    model = model.drop(columns=vars_to_drop)
    # Dummy variable categorical variables
    model = pd.get_dummies(data=model, columns=cat_vars_to_model, drop_first=False)
    # Create target variable
    y = model['num']
    # Create feature variables
    x = model.drop(columns='num')

    # Create copy of x for standard scaling
    x_std = x.copy()
    # print(list(x_std)[list(x_std).index('sex_0')-1])
    x_std.loc[:, :list(x_std)[list(x_std).index('sex_0')-1]] = scaler.fit_transform(x_std.loc[:, :list(x_std)[list(x_std).index('sex_0')-1]])

    # Define parameters of SVC
    svc_model = SVC(kernel='linear')
    # Recursive feature elimination
    rfe_svc = pd.DataFrame(data=[list(x_std), RFE(svc_model, n_features_to_select=1).fit(x_std, y).ranking_.tolist()]).T.\
        rename(columns={0: 'variable', 1: 'rfe_ranking'}).sort_values(by='rfe_ranking').reset_index(drop=True)
    svc_model = SVC(random_state=1)
    param_grid = {'kernel': ['rbf', 'sigmoid', 'linear'], 'C': np.arange(0.10, 2.41, step=0.05), 'gamma': ['scale', 'auto']}
    cv = ShuffleSplit(n_splits=5, test_size=0.3)
    # Define grid search CV parameters
    grid_search = GridSearchCV(svc_model, param_grid, cv=cv)

    # Loop through features based on recursive feature elimination evaluation - top to bottom
    model_search_svc = []
    svc_variable_list = []
    for i in range(len(rfe_svc)):
        if rfe_svc['variable'][i] not in svc_variable_list:
            svc_variable_list.extend([rfe_svc['variable'][i]])
            # Add related one-hot encoded variables if variable is categorical
            if svc_variable_list[-1].split('_')[-1] in sorted([x for x in list(set([x.split('_')[-1] for x in list(x_std)])) if len(x) == 1]):
                svc_variable_list.extend([var for var in list(x_std) if svc_variable_list[-1].split('_')[0] in var and var != svc_variable_list[-1]])
            print(svc_variable_list)
            grid_search.fit(x_std[svc_variable_list], y)
            print(f'Best parameters for current grid seach: {grid_search.best_params_}')
            print(f'Best score for current grid seach: {grid_search.best_score_}')
            # Define parameters of Support-vector machine classifer from grid search
            svc_model = SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'],
                            gamma=grid_search.best_params_['gamma'], random_state=1)
            # Cross-validate and predict using Support-vector machine classifer
            svc_predict = cross_val_predict(svc_model, x_std[svc_variable_list], y, cv=5)
            print(confusion_matrix(y_true=y, y_pred=svc_predict))
            conf_matr = confusion_matrix(y_true=y, y_pred=svc_predict)
            model_search_svc.append([grid_search.best_params_, grid_search.best_score_, conf_matr[0][0],
                                     conf_matr[0][1], conf_matr[1][0], conf_matr[1][1], list(x_std[svc_variable_list])])
    # Create DataFrame of svc results
    model_search_svc = pd.DataFrame(model_search_svc, columns=['best_model_params_grid_search', 'best_score_grid_search',
                                                 'true_negatives', 'false_positives',
                                                 'false_negatives', 'true_positives', 'variables_used'])
    # Create recall, precision, f1-score columns
    model_search_svc['recall'] = model_search_svc.true_positives/(model_search_svc.true_positives + model_search_svc.false_negatives)
    model_search_svc['precision'] = model_search_svc.true_positives/(model_search_svc.true_positives + model_search_svc.false_positives)
    model_search_svc['f1_score'] = 2 * (model_search_svc.precision * model_search_svc.recall) / (model_search_svc.precision + model_search_svc.recall)
    # Sort DataFrame
    model_search_svc = model_search_svc.sort_values(by=['f1_score'], ascending=False)
    print(model_search_svc)

    # Choose top model from svc model search
    if len(model_search_svc.loc[model_search_svc.f1_score==model_search_svc.f1_score.max()]) > 1:
        top_model_result_svc = model_search_svc.loc[(model_search_svc.f1_score == model_search_svc.f1_score.max()) &
            (model_search_svc['variables_used'].apply(len) == min(map(lambda x: len(x[list(model_search_svc).index('variables_used')]),
             model_search_svc.loc[model_search_svc.f1_score==model_search_svc.f1_score.max()].values)))]
        if len(top_model_result_svc) > 1:
            print('break here')
            break
            # top_model_result_svc = top_model_result_svc.loc[top_model_result_svc.best_score_grid_search == top_model_result_svc.best_score_grid_search.max()]
    else:
        top_model_result_svc = model_search_svc.loc[model_search_svc.f1_score==model_search_svc.f1_score.max()]
    top_model_results = top_model_results.append(other=top_model_result_svc, sort=False)
    print(f"Top svc model: \n {top_model_result_svc}")

    # Append top_model_result_svc results to all_model_results DataFrame
    all_model_results['svc_'+inflect.engine().number_to_words(index)] = cross_val_predict(SVC(
        kernel=top_model_result_svc['best_model_params_grid_search'].values[0]['kernel'],
        C=top_model_result_svc['best_model_params_grid_search'].values[0]['C'],
        gamma=top_model_result_svc['best_model_params_grid_search'].values[0]['gamma']),
        x_std[top_model_result_svc['variables_used'].values[0]], y, cv=5)

# Fill in model_type columns
top_model_results['model_type'] = top_model_results['model_type'].fillna(value='svc')
```

K-Nearest Neighbors
```python
# Unique variable combination runs
for index, (vars_to_drop, cat_vars_to_model) in enumerate(zip(variables_to_drop_list,
                                                              categorical_variables_for_modeling_list), start=1):
    print(f"Model run: {index}")
    # Create copy of hungarian for non-regression modeling
    model = hungarian.copy()
    # Drop columns
    model = model.drop(columns=vars_to_drop)
    # Dummy variable categorical variables
    model = pd.get_dummies(data=model, columns=cat_vars_to_model, drop_first=False)
    # Create target variable
    y = model['num']
    # Create feature variables
    x = model.drop(columns='num')

    # Create copy of x for standard scaling
    x_std = x.copy()
    x_std.loc[:, :list(x_std)[list(x_std).index('sex_0')-1]] = scaler.fit_transform(x_std.loc[:, :list(x_std)[list(x_std).index('sex_0')-1]])

    # Use Recursive Feature Elimination from SVC
    # Define parameters of SVC
    svc_model = SVC(kernel='linear', random_state=1)
    # Feature importance DataFrame
    feature_info = pd.DataFrame(data=[list(x_std), RFE(svc_model, n_features_to_select=1).fit(x_std, y).ranking_.tolist()]).T.\
        rename(columns={0: 'variable', 1: 'rfe_svc'}).reset_index(drop=True)

    # Define parameters of Random Forest Classifier
    random_forest_model = RandomForestClassifier(random_state=1)
    # Merge feature importances from random forest classifer on feature_info
    feature_info = feature_info.merge(pd.DataFrame(data=[list(x), random_forest_model.fit(x,y).feature_importances_.tolist()]).T.\
        rename(columns={0: 'variable', 1: 'feature_importance_rfc'}), on='variable')
    # Sort values by descending random forest classifier feature importance to create ranking column
    feature_info = feature_info.sort_values(by='feature_importance_rfc', ascending=False)
    feature_info['feature_importance_rfc_ranking'] = np.arange(1,len(feature_info)+1)

    # Define parameters of Gradient Boosting Classifier
    gbm_model = GradientBoostingClassifier(random_state=1)
    # Merge feature importances from gradient boosting classifer on feature_info
    feature_info = feature_info.merge(pd.DataFrame(data=[list(x), gbm_model.fit(x,y).feature_importances_.tolist()]).T.\
        rename(columns={0: 'variable', 1: 'feature_importance_gbm'}), on='variable')
    # Sort values by descending gradient boosting classifier feature importance to create ranking column
    feature_info = feature_info.sort_values(by='feature_importance_gbm', ascending=False)
    feature_info['feature_importance_gbm_ranking'] = np.arange(1,len(feature_info)+1)

    # Get average of three RFE/feature importance columns
    feature_info['feature_importance_avg'] = feature_info[['rfe_svc', 'feature_importance_rfc_ranking', 'feature_importance_gbm_ranking']].mean(axis=1)
    # Sort values by average column
    feature_info = feature_info.sort_values(by='feature_importance_avg', ascending=True).reset_index(drop=True)

    # Define parameters of kNN model
    knn_model = KNeighborsClassifier(metric='minkowski')
    # Define parameters of grid search
    param_grid = {'n_neighbors': np.arange(9, 47, step=2), 'weights': ['uniform', 'distance']}
    # Define parameters of shuffle split
    cv = ShuffleSplit(n_splits=5, test_size=0.3)
    # Define grid search CV parameters
    grid_search = GridSearchCV(knn_model, param_grid, cv=cv) # , scoring='recall'
    # Append model results to this list
    model_search_knn = []
    # Begin top to bottom process - looking at most important variables (by RFE ranking first and adding on)
    knn_variable_list = []
    for i in range(len(feature_info)):
        if feature_info['variable'][i] not in knn_variable_list:
            knn_variable_list.extend([feature_info['variable'][i]])
            # Add related one-hot encoded variables if variable is categorical
            if knn_variable_list[-1].split('_')[-1] in sorted([x for x in list(set([x.split('_')[-1] for x in list(x_std)])) if len(x) == 1]):
                knn_variable_list.extend([var for var in list(x_std) if knn_variable_list[-1].split('_')[0] in var and var != knn_variable_list[-1]])
            print(knn_variable_list)
            grid_search.fit(x_std[knn_variable_list], y)
            print(f'Best parameters for current grid seach: {grid_search.best_params_}')
            print(f'Best score for current grid seach: {grid_search.best_score_}')
            # Define parameters of k-nearest neighbors from grid search
            knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=grid_search.best_params_['n_neighbors'],
                            weights=grid_search.best_params_['weights'])
            # Cross-validate and predict using Support-vector machine classifer
            knn_predict = cross_val_predict(knn_model, x_std[knn_variable_list], y, cv=5)
            print(confusion_matrix(y_true=y, y_pred=knn_predict))
            conf_matr = confusion_matrix(y_true=y, y_pred=knn_predict)
            model_search_knn.append([grid_search.best_params_, grid_search.best_score_, conf_matr[0][0],
                                     conf_matr[0][1], conf_matr[1][0], conf_matr[1][1], list(x_std[knn_variable_list])])
    # Create DataFrame of k-nearest neighbors results
    model_search_knn = pd.DataFrame(model_search_knn, columns=['best_model_params_grid_search', 'best_score_grid_search',
                                                 'true_negatives', 'false_positives',
                                                 'false_negatives', 'true_positives', 'variables_used'])
    # Create recall, precision, f1-score columns
    model_search_knn['recall'] = model_search_knn.true_positives/(model_search_knn.true_positives + model_search_knn.false_negatives)
    model_search_knn['precision'] = model_search_knn.true_positives/(model_search_knn.true_positives + model_search_knn.false_positives)
    model_search_knn['f1_score'] = 2 * (model_search_knn.precision * model_search_knn.recall) / (model_search_knn.precision + model_search_knn.recall)
    # Sort DataFrame
    model_search_knn = model_search_knn.sort_values(by=['f1_score'], ascending=False)
    print(model_search_knn)

    if len(model_search_knn.loc[model_search_knn.f1_score==model_search_knn.f1_score.max()]) > 1:
        print("Fix multiple best model problem for rfc")
        break
    else:
        top_model_result_knn = model_search_knn.loc[model_search_knn.f1_score==model_search_knn.f1_score.max()]
    top_model_results = top_model_results.append(other=top_model_result_knn, sort=False)
    print(f"Top knn model: \n {top_model_result_knn}")

    # Append top_model_result_knn results to all_model_results DataFrame
    knn_predict_proba = cross_val_predict(KNeighborsClassifier(metric='minkowski',
                               n_neighbors=top_model_result_knn['best_model_params_grid_search'].values[0]['n_neighbors'],
                               weights=top_model_result_knn['best_model_params_grid_search'].values[0]['weights']),
                     x_std[top_model_result_knn['variables_used'].values[0]], y, cv=5, method='predict_proba')
    all_model_results['knn_'+inflect.engine().number_to_words(index)+'_pred_zero'] = knn_predict_proba[:,0]
    all_model_results['knn_'+inflect.engine().number_to_words(index)+'_pred_one'] = knn_predict_proba[:,1]

# Fill in model_type columns
top_model_results['model_type'] = top_model_results['model_type'].fillna(value='knn')
```


(Put code at bottom - base off table of contents and say for all code (script) - go to the Github page for the project (give link to heart disease))

Questions: how else could the model be useful? Other visuals that could be useful to visualize the results? Any strategies that could make the model more useful? Feature engineering ideas? Any questions or lack of understanding on code?
