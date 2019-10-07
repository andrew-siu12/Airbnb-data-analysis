# Airbnb Data Analysis

## Introduction

Airbnb is one of the most popular online community marketplace for people ('hosts) to list properties, book experiences and discover places. Hosts are reponsible for setting prices for the listings. It is hard for newcomers to set an accurate price to compete with other experience hosts.

As one of the most popular cities in Europe, London has over 80,000 listings as of May 2019. In such fierce competition environments, it is important to know which factors driving the price of listings. In this post, we will perform data analysis to extract useful insights about rental landscape in London. 
And applying machine learning models to predict the price for listings in London. The questions that will be addressed in this project are:

1. How is the demand for Airbnb changing in London?
2. Which month is more expensive to travel on?
3. Whcih boroughs are more expensive,  and which areas have the best reviews?
4. What are the factors that influence pricing on Airbnb listings?

Random Forests and LightGBM were trained to predict the price of the listings in London.

## Data 
[Inside Airbnb](http://insideairbnb.com/get-the-data.html) has provided data that is sourced from public available infomration from Airbnb webiste. The data we used for this project is compiled on 05 May, 2019. The dataset comprised of three tables and a geojson file of London boroughs:
* `listings` - Deatailed listings data for London
* `calendar` - Deatailed bookings for the next calendar year of listings
* `reviews` - Detailed reviews data for listings in London.
* `neigbourhoods` - geojson file of boroughs of London.

## File Descriptions
```
.
├── images
├── models
│   └── bayes_test.csv
├── notebook
│   ├── Airbnb\ Review\ data.ipynb
│   ├── Airbnb\ calendar\ data.ipynb
│   ├── Airbnb-data-analysis.ipynb
│   └── Hyperparameter\ tuning.ipynb
├── preprocessed_data

└── src
    ├── __pycache__
    │   ├── const.cpython-37.pyc
    │   ├── feature_extraction.cpython-37.pyc
    │   ├── util.cpython-37.pyc
    │   └── utils.cpython-37.pyc
    ├── const.py
    ├── feature_extraction.py
    ├── final_model.py
    ├── model_tuning.py
    ├── pipeline.py
    └── util.py


```
* `models` - this folder contains the tuning hyperparameters stored in csv file
* `notebook` - Contains EDA of review data, calendar data and pricing data, and hyperparameter tuning notebook
* `src` - Pipeline script, prediction script and tuning script


## Conclusions
* The demand for Airbnb increased exponentially over the years. 
* Demand increases along the year until August and slowly decrease.
* The cheapest month to book Airbnb is May.
* Richmond upon Thames borough has the highest average review score ratings.
* Our best model to predict the price is using LightGBM tuned with hyperopts which has an adjusted r2 score of 76.923%. This means that the model explain 76.923% of the variability in listing price. The result was not good enough. However, price is very difficult to predict correcly, and the remaining 23.077% may be explained by features that are not used in the model. 

A summary post has been written on my own [blog](https://andrew-siu12.github.io/2019-06-15-Airbnb-Data-Analysis/)

