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

## Data Ingestion
The first step was obtaining the [data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data) from the UCI Machine Learning Repository. The data was then ingested into Python.
  
## Data Cleaning  
After the data was properly read into into Python and the appropriate column names were supplied, data cleaning was performed. This involved: 
* Removing unnecessary columns
* Removing rows with a large percentage of missing values
* Imputing missing values for patients using K-Nearest Neighbors, an advanced data imputation method  

## Exploratory Data Analysis
The following two images provide a sample of the analysis performed.

![Distribution_of_Continuous_Features_by_Target](distribution_of_continuous_features_by_target.png "Distribution of Continuous Features by Target")


Hi there! I'm Paul. I’m a physics major turned programmer. Ever since I first learned how to program while taking a scientific computing for physics course, I have pursued programming as a passion, and as a career. Check out [my personal website](https://www.lenpaul.com/) for more information on my other projects (including more Jekyll themes!), as well as some of my writing.
