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
* Data Visualization
* Data Transformations (add section for this? visualizaton showing the differences - chol) [<sub><sup>View code</sup></sub>](#c)

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

DataFrame of continuous variable correlations greater than 0.6 and less than 1.0 and less than -0.6 and greater than -1.0 with all null columns dropped - with all feature engineered variables included
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



(Put code at bottom - base off table of contents and say for all code (script) - go to the Github page for the project (give link to heart disease))

Questions: how else could the model be useful? Other visuals that could be useful to visualize the results? Any strategies that could make the model more useful? Feature engineering ideas? Any questions or lack of understanding on code?
