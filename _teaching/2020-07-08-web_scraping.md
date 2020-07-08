---
layout: post
title: "Web Scraping using Selenium with Python"
author: "Dustin Wicker"
categories: journal
tags: [automobiles,cars,web scraping,data science,data analysis,selenium,python]
image: ford_mustang.jpg
---

Web scraping, the act of extracting data from websites, is an extremely useful tool for data scientists in today's data-packed world.
* Being able to effectively web scrape means the **_internet is your database_**
* The technique can be utilized in multiple different ways, ranging from scraping pertinent data from COVID-19 related websites to help fellow volunteers create an [altruistic meta-platform](https://fightpandemics.com/) that brings people and organizations together to selling items programmatically on Poshmark to scraping Google Search results.  
  
This tutorial details how to pull 2020 Mustang information from Ford's site using a popular open-source web-based automation tool called [Selenium](https://selenium-python.readthedocs.io/).
* Using the data scraped from this tutorial would allow you to compare the price and lease amount per month to the city and highway miles per gallon information of 2020 Ford Mustang's to get the most economical and fuel-efficient car.
* Using (and expanding) on this tutorial to extract information on all 2020 Ford models, plus other car companies such as Toyota and Chevrolet, would allow you to build a recommendation system that helps people find the right car for their preferences.

Run the code as you are following along. This will ensure you understand each piece of the process, and help you gain the knowledge to web scrape on your own. If you are interested in viewing and obtaining the full script, please visit this [link](https://github.com/dustinwicker/Thinkful/blob/master/car_scrape.py).

The first step is **importing necessary libraries and packages and setting display options to make our output easier to read**.
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
  
Now it is time to **set WebDriver options, define the WebDriver, and load the WebDriver in our current browser session**.
* The **'--headless' argument** runs the web browser with **no user interface**; it essentially allows the browser to operate in the background without any pop-up window. This is a great tool to use once you feel comfortable web scraping, and want to automate a task without starting up the user interface of the browser. For now we will comment the argument out, but feel free to use it once you have run the code a few times from top to bottom and feel ready.
```python
# Webdriver options
options = Options()
# # Make call to Chrome
# options.add_argument('--headless')
# Define Chrome webdriver for site
driver = webdriver.Chrome(options=options)
# Define url
url = "https://www.ford.com/cars/mustang/models/"
# Supply url
driver.get(url=url)
```  
  
To **find the necessary information needed to extract the data**, you have a few options.  
  
**Option 1**: 
* Right-click on the webpage and select 'Inspect'  
  
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_6.png "Distributions of Continuous Features by Target")  
  

**Option 2**:  
* Click on the three vertical dots in the upper right-hand corner  
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_1.png "Distributions of Continuous Features by Target")  
  
* With the panel open, come down and click or hover over 'More Tools'  
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_2.png "Distributions of Continuous Features by Target")  

* Clicking over hovering over 'More Tools' will open up another panel - in this new panel, click on 'Developer Tools'
   * Notice there is keyboard shortcut available as well for quicker access
 ![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_4.png "Distributions of Continuous Features by Target")  

**Option 3**:
* Use a keyboard shortcut
   * Mac: Command+Option+C
   * Windows/Linux: Control+Shift+C  
  
  
All three of these options will open up the **Elements panel where the DOM (Document Object Model) can be inspected - this is the information we will use to web scrape**.  
  
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_5.png "Distributions of Continuous Features by Target")  
  
It is now time to **determine the necessary element or elements that contain the car information we need**.  
* Head to the Elements panel, and click on the icon in the upper left-hand corner. This will allow you to select a specific element on the page, and inspect it.
![Distribution_of_Continuous_Features_by_Target](/assets/img/visual_guide_to_get_scraping_info_7.png "Distributions of Continuous Features by Target")  
  
  
* Put your cursor around one of the cars so you obtain, via the colored rectangles, the car's picture, starting price, miles per gallon, and leasing price information as demonstrated in the image below.  
  
* Click and the Elements panel will highlight the particular element of interest. In this case, it's a div element with a class of wrap.
![test1](/assets/img/visual_guide_to_get_scraping_info_8.png "test1")              ![test2](/assets/img/visual_guide_to_get_scraping_info_9.png "test2")  
  
* Shifting focus to the Elements panel above on the right, notice the **"START VEHICLE TILE" comment**.
   * Given that a div element with a class equal to "wrap" is likely not unique, pulling something _more specific_ will help us be certain we are obtaining the correct results.
   * In the Elements panel, move the cursor up to the div element with the class "vehicleTile section.
   * All the information we need to web scrape is contained in the rectangle created by hovering over that div. We have located the correct element we need.  
  
Here is how we will **use that information to extract the data**.
```python
# Obtain vehicle information for each of the displayed Mustangs
cars = driver.find_elements_by_xpath("//div[@class='vehicleTile section']")
```

**Lets observe the web elements obtained**.
```python
cars

Out[1]:
[<selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-1")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-2")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-3")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-4")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-5")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-6")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-7")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-8")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-9")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-10")>,
 <selenium.webdriver.remote.webelement.WebElement (session="3b276f6f5df22154cb9b8fb2141eb262", element="0.8182600666874935-11")>]
 ```
 
**Lets inspect the first car to see the information we scraped for it**.
* Use ```.text``` on the web element to return the text within the element.
   
```python
# Lets observe the first car
first_car = cars[0].text
# Print first_car
first_car

Out[2]:
'2020 MUSTANG ECOBOOST® FASTBACK\nStarting at $26,670 1 \nEPA-Est. MPG 21 City / 31 HWY 2 \nLease at $315/mo 7 \nPress here for information on 2020 Ford Mustang monthly pricing'
```

```python
# Print type of first_car
type(first_car)

Out[3]:
str
```

```python
# Lets split on the new line (\n) since it separates the various pieces of information of interest
first_car = first_car.split("\n")
# Print type
type(first_car)

Out[4]: list
```

```python
# Lets get all of our desired information from the first car - year, car model, price, city mpg, hwy mpg, lease/mo
# year
first_car[0][0:4]

Out[5]: '2020'

# car model
first_car[0][4:].strip()

Out[6]: 'MUSTANG ECOBOOST® FASTBACK'


# price
# Locate the position of the '$'
first_car[1].index('$')
# Use the position of the $ to find the full price
first_car[1][first_car[1].index('$'): first_car[1].index('$')+7]
# Remove the $ and , so we can get the integer value
first_car[1][first_car[1].index('$'): first_car[1].index('$')+7].replace("$", "").replace(",", "")

Out[7]: '26670'


# city mpg
first_car[2].split("/")[0].split('MPG')[1].strip()[0:2]

Out[8]: '21'


# hwy mpg
first_car[2].split("/")[1].strip().split('HWY')[0].strip()

Out[9]: '31'


# lease/mo
# Locate the position of the '$' and use the position of the $ to find the lease/mo amount
first_car[3][first_car[3].index('$'):first_car[3].index('$')+4]
# Remove the $ so we can get the integer value
first_car[3][first_car[3].index('$'):first_car[3].index('$')+4].replace("$", "")

Out[10]: '315
```

**Lets apply the above methodologies to all cars extracted from the website to get their specific information**.
```python
# Use an empty list to capture the results
car_results_list = []
for i in range(len(cars)):
    # Use this list to capture the results of each car. After each for loop run, we append these results to our other
    # 'main' list (car_results_list)
    specific_car_info_list = []
    car = cars[i].text.split("\n")
    # year
    specific_car_info_list.extend([car[0][0:4]])
    # car model
    specific_car_info_list.extend([car[0][4:].strip()])
    # price
    specific_car_info_list.extend([car[1][car[1].index('$'): car[1].index('$') + 7].replace("$", "").replace(",", "")])
    # The following try/except blocks allow us to check if that car listing has the information we want to capture.
    # If it does not (i.e., we hit the except part, we extend an nan onto that car's specific information list)
    try:
        specific_car_info_list.extend([car[2].split("/")[0].split('MPG')[1].strip()[0:2]])  # city mpg
    except IndexError:
        print('No city mpg for this vehicle.')
        specific_car_info_list.extend([np.nan])
    try:
        specific_car_info_list.extend([car[2].split("/")[1].strip().split('HWY')[0].strip()])  # hwy mpg
    except:
        print('No hwy mpg for this vehicle.')
        specific_car_info_list.extend([np.nan])
    try:
        specific_car_info_list.extend([car[3][car[3].index('$'):car[3].index('$') + 4].replace("$", "")])  # lease/mo
    except:
        print('No lease/mo for this vehicle.')
        specific_car_info_list.extend([np.nan])
    car_results_list.append(specific_car_info_list)
 ```
 
Now we have a **list of lists** - each specific list contains information for one car.
 ```python
 print(car_results_list)
 
 Out[11]: 
[['2020', 'MUSTANG ECOBOOST® FASTBACK', '26670', '21', '31', '315'],
 ['2020', 'MUSTANG ECOBOOST® PREMIUM FASTBACK', '31685', '21', '31', '374'],
 ['2020', 'MUSTANG ECOBOOST® CONVERTIBLE', '32170', '20', '28', '412'],
 ['2020', 'MUSTANG GT FASTBACK', '35880', '15', '25', '502'],
 ['2020', 'MUSTANG ECOBOOST® PREMIUM CONVERTIBLE', '37185', '20', '28', '476'],
 ['2020', 'MUSTANG GT PREMIUM CONVERTIBLE', '45380', '15', '25', '631'],
 ['2020', 'MUSTANG GT PREMIUM FASTBACK', '39880', '15', '25', '557'],
 ['2020', 'MUSTANG BULLITT™', '47705', '15', '25', '663'],
 ['2020', 'MUSTANG SHELBY GT350®', '60440', '14', '21', nan],
 ['2020', 'MUSTANG SHELBY® GT350R', '73435', '14', '21', nan],
 ['2020', 'MUSTANG SHELBY® GT500®', '72900', nan, nan, nan]]


# Notice the first element in the list corresponds to the first car
car_results_list[0]

Out[12]: ['2020', 'MUSTANG ECOBOOST® FASTBACK', '26670', '21', '31', '315']
```  

**Convert the list of lists to a DataFrame**.  
```python
cars_df = pd.DataFrame(data=car_results_list, columns=['year', 'car_name', 'price', 'city_mpg', 'hwy_mpg', 'lease_mo'])
print(cars_df)

Out[13]:
    year                               car_name  price city_mpg hwy_mpg lease_mo
0   2020  MUSTANG ECOBOOST® FASTBACK             26670  21       31      315    
1   2020  MUSTANG ECOBOOST® PREMIUM FASTBACK     31685  21       31      374    
2   2020  MUSTANG ECOBOOST® CONVERTIBLE          32170  20       28      412    
3   2020  MUSTANG GT FASTBACK                    35880  15       25      502    
4   2020  MUSTANG ECOBOOST® PREMIUM CONVERTIBLE  37185  20       28      476    
5   2020  MUSTANG GT PREMIUM CONVERTIBLE         45380  15       25      631    
6   2020  MUSTANG GT PREMIUM FASTBACK            39880  15       25      557    
7   2020  MUSTANG BULLITT™                       47705  15       25      663    
8   2020  MUSTANG SHELBY GT350®                  60440  14       21      NaN    
9   2020  MUSTANG SHELBY® GT350R                 73435  14       21      NaN    
10  2020  MUSTANG SHELBY® GT500®                 72900  NaN      NaN     NaN 
```

**As a final data cleaning step, lets remove all non-ASCII characters to make the strings easier to work with**.
```python
# Dictionary to detect non-ASCII characters
find = dict.fromkeys(range(128), '')

# Get a list of all non-ascii characters in car_name column
non_ascii_characters_list = []
for car in cars_df['car_name']:
    for letter in list(car):
        if letter.translate(find):
            if letter.translate(find) not in non_ascii_characters_list:
                non_ascii_characters_list.append(letter.translate(find))

for symbol in non_ascii_characters_list:
    for index, value in enumerate(cars_df['car_name']):
        cars_df.loc[index]['car_name'] = cars_df.loc[index]['car_name'].replace(symbol, '')
        
print(cars_df)

Out[14]: 
    year                              car_name  price city_mpg hwy_mpg lease_mo
0   2020  MUSTANG ECOBOOST FASTBACK             26670  21       31      315    
1   2020  MUSTANG ECOBOOST PREMIUM FASTBACK     31685  21       31      374    
2   2020  MUSTANG ECOBOOST CONVERTIBLE          32170  20       28      412    
3   2020  MUSTANG GT FASTBACK                   35880  15       25      502    
4   2020  MUSTANG ECOBOOST PREMIUM CONVERTIBLE  37185  20       28      476    
5   2020  MUSTANG GT PREMIUM CONVERTIBLE        45380  15       25      631    
6   2020  MUSTANG GT PREMIUM FASTBACK           39880  15       25      557    
7   2020  MUSTANG BULLITT                       47705  15       25      663    
8   2020  MUSTANG SHELBY GT350                  60440  14       21      NaN    
9   2020  MUSTANG SHELBY GT350R                 73435  14       21      NaN    
10  2020  MUSTANG SHELBY GT500                  72900  NaN      NaN     NaN
```
