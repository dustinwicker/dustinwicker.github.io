---
layout: post
title: "Predicting the Presence of Heart Disease in Patients"
author: "Dustin Wicker"
categories: journal
tags: [healthcare,data science,data analytics,data analysis,machine learning,sample]
image: bar_chart_confusion_matrix_svc.png
---

## Project Summary  
Statistical analysis, data mining techniques, and five machine learning models (include the five models?) were built and ensembled to accurately predict the presence of heart disease in patients from the Hungarian Institute of Cardiology in Budapest. All told, the model which provided the optimal combination of total patients predicted correctly and F1 Score, while being the most parsimonious one, was the Support Vector Machine Classification Model #4 (bold important words.)

 (Put overall summary of model results)
 
# Project Overview  
## [Data Ingestion](#data-ingestion)
## [Data Cleaning](#data-cleaning)
## [Exploratory Data Analysis](#exploratory-data-analysis)
## [Model Building](#model-building)

## Data Ingestion
The first step was obtaining the [data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data) from the UCI Machine Learning Repository. The data was then ingested into Python.

(Explain the data set - what target is)
  
## Data Cleaning  
After the data was properly read into into Python and the appropriate column names were supplied, data cleaning was performed. This involved: 
* Removing unnecessary columns
* Removing rows with a large percentage of missing values
* Imputing missing values for patients using K-Nearest Neighbors, an advanced data imputation method  

## Exploratory Data Analysis
The following two images provide a sample of the analysis performed.

![Distribution_of_Continuous_Features_by_Target](/assets/img/distribution_of_continuous_features_by_target.png "Distribution of Continuous Features by Target")

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







(Put code at bottom - base off table of contents and say for all code (script) - go to the Github page for the project (give link to heart disease))


Hi there! I'm Paul. I’m a physics major turned programmer. Ever since I first learned how to program while taking a scientific computing for physics course, I have pursued programming as a passion, and as a career. Check out [my personal website](https://www.lenpaul.com/) for more information on my other projects (including more Jekyll themes!), as well as some of my writing.
