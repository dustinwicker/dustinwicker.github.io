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
The first step was obtaining the [data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data) from the UCI Machine Learning Repository. The file was saved in an appropriate location on my machine and then read into Python. [<sub><sup>View code</sup></sub>](#a)

## Data Cleaning  
After the data was properly read into into Python and the appropriate column names were supplied, data cleaning was performed. This involved: 
* Removing unnecessary columns
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



(Put code at bottom - base off table of contents and say for all code (script) - go to the Github page for the project (give link to heart disease))

Questions: how else could the model be useful? Other visuals that could be useful to visualize the results? Any strategies that could make the model more useful? Feature engineering ideas? Any questions or lack of understanding on code?


Hi there! I'm Paul. I’m a physics major turned programmer. Ever since I first learned how to program while taking a scientific computing for physics course, I have pursued programming as a passion, and as a career. Check out [my personal website](https://www.lenpaul.com/) for more information on my other projects (including more Jekyll themes!), as well as some of my writing.
