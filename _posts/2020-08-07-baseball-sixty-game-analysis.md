---
layout: post
title: "Examining the Effectiveness of MLB's 2020 Sixty Game Baseball Season using Python"
author: "Dustin Wicker"
categories: journal
tags: [data science,data analysis,statistics,hypothesis testing,data visualization,data mining,data cleaning,web scraping,python]
image: atlanta_braves.jpg
---

*** UNDER CONSTRUCTION ***

On June 23, 2020, Major League Baseball (MLB) announed they were officially coming back and instituting a sixty-game baseball season for the 2020 campaign. Baseball fans everywhere rejoiced; excited to be able to watch the sport they love again, and get a reprieve, albeit temporary, from the mundane lifestyle set in by the coronavirus pandemic. Since MLB's announcement, I have found myself contemplating how worthwile this shortened season will be. Will this season go down in the hollowed record books as being as meaningful as those of past seasons, or will the proverbial baseball asterick be placed next to it and it's outcomes? I compiled data for the last twenty-five baseball seasons (1995 - 2019), and analyzed it using statisical methods and visualizations to answer this question. Follow along to determine if you should celebrate (socially distanced of course) 


To answer this question,  to determine if the teams in playoff contention at the sixty-game mark were the same teams that ended up making the playoffs at season's end. 


 Please leave a comment at the bottom of the page to let me know your thoughts on the project and the actions I took to arrive at the final model. At the end of the [Project Overview](#project-overview), there are questions posed for reflection and deliberation - feel free to answer one of those too if you would like.

## Project Summary  

Code snippets will be provided for each section outlined in the [Project Overview](#project-overview) at the bottom of the page. The snippets will encompass the entire script, broken into their related sections. If you would like to view the code script in its entirety, please visit this [link](https://github.com/dustinwicker/Heart-Disease-Detection/blob/master/heart_disease_code.py/?target=%22_blank%22).
 
# Project Overview  
## i.    [Data Ingestion](#data-ingestionview-code)
## ii.   [Data Cleaning](#data-cleaningview-code)
## iii.  [Exploratory Data Analysis](#exploratory-data-analysisview-code)
## iv.  [Model Building](#model-buildingview-code)
## v.   [Model Visualization, Comparison, and Selection](#model-visualization-comparison-and-selectionview-code)
## vi.  [Visualize Best Model](#visualize-best-modelview-code)
## vii. [Model Usefulness](#model-usefulness)
## viii.[Questions for Consideration](#questions-for-consideration) 
  
## Data Ingestion [<sub><sup>(View code)</sup></sub>](#data-ingestion)  

## Data Cleaning [<sub><sup>(View code)</sup></sub>](#data-cleaning)


## Exploratory Data Analysis [<sub><sup>(View code)</sup></sub>](#exploratory-data-analysis)

![Heatmap of Continous Predictor Variables](/assets/img/heatmap_continous_predictor_variables.png "Heatmap of Continous Predictor Variables")

   
## Model Visualization, Comparison, and Selection [<sub><sup>(View code)</sup></sub>](#model-visualization-comparison-and-selection)
* ROC Curves were built based on each model's predicted probabilities to visually compare model performance at various cut-off values.  

![ROC Curves](/assets/img/roc_cruves.png "ROC Curves")  

## Questions for Consideration/Analysis Expansion in the Future
* Pull all teams season ending records and statistics (correlations between 60 game playoff teams and their final result and 60 game standings and playoff teams - their correlations) .54
