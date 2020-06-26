---
layout: post
title: "Web Scraping"
author: "Dustin Wicker"
categories: journal
tags: [documentation,sample]
image: spools.jpg
---

Web scraping, the act of extracting data from websites, is an extremely useful tool for a data scientist in today's data-packed world.
* Being able to effectively web scrape means the _**internet is your database**_
* The technique can be utilized in multiple different ways, ranging from scraping pertinent data from COVID-19 related websites to help fellow volunteers create an altruistic meta-platform (link to FightPandemics) that brings people and organizations together to selling items on Poshmark (put link in for post to this) to scraping Google Search results. (to decipher trends and patterns?)  
  
This tutorial details how to pull 2020 Mustang information from Ford's site using a popular open-source web-based automation tool called Selenium (link).

Run the code as you're following along and also give link to Google Colab file on Github.

The first step will be importing necessary libraries and packages and setting display options to make our output easier to read.
```python
# Import libraries and packages
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Be able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', -1)
# Increase number of rows printed out in console
pd.set_option('display.max_rows', 200)
```  
  
Now it is time to set WebDriver options, define the WebDriver, and then load it in our current brower session
* The '--headless' argument runs the web browser with no user interface; it essentially allows the browser to operate in the background without any pop-up window. This is a great tool to use once you feel comfortable web scraping and want to automate a task without starting up the user interface of the browser.
```python
# Webdriver options
options = Options()
# # Make call to Chrome headless
# options.add_argument('--headless')
# Define Chrome webdriver for site
driver = webdriver.Chrome(options=options)
# Define url
url = "https://www.ford.com/cars/mustang/models/"
# Supply url
driver.get(url=url)
```  
  
To find the necessary information needed to extract the data:  
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_1.png "Distributions of Continuous Features by Target")
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_2.png "Distributions of Continuous Features by Target")
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_4.png "Distributions of Continuous Features by Target")
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_5.png "Distributions of Continuous Features by Target")


