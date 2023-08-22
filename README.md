# GetAround Car Rental Service Project 🚗 🚗 🚗

## Overview 🌐
GetAround is an online car rental service operating much like Airbnb for cars. Founded in 2009, the platform has experienced rapid growth, boasting over 5 million users and about 20,000 available cars worldwide by 2019. Users must complete a check-in and check-out flow at the beginning and end of each rental for several purposes:
1.Assess the condition of the car and report any pre-existing or new damages. 
2.Compare fuel levels before and after rental. 
3.Measure the distance driven during the rental period.

## Project Goals 🎯

# Primary Goals
Implement Minimum Delay: To address issues related to late returns, a **minimum delay** will be set between two consecutive rentals for the same car. Decisions Pending: Minimum delay duration, scope of application (all cars, only Connect cars, etc.) Rental Price Prediction: Provide an API with a /predict endpoint that estimates the rental price per day.

# Secondary Goals
Analyze the impact of implementing the minimum delay feature on owner's revenue and user experience. Determine the frequency and impact of late check-ins and checkouts.

## API 💻
This is an app which **predicts car rental prices** based on certain parameters. These are the parameters : car model's name, mileage, engine_power, fuel, car_type, as well an aviability of following features : private parking, gps, has_air_conditioning, automatic_car, has_getaround_connect, has_speed_regulator, winter_tires.
User can either fill in the fields or make request by posting data in json format.
Link to API documentation : 
https://priceestimatorsuperapp-4183505f60be.herokuapp.com/docs

## Streamlit Dashboard 📊
Dashboard wich explores the data and helps to get useful insights in order to make a right descision. Link to the dashboard : 
https://dashdahscars-a74765b4f79d.herokuapp.com/

## Machine Learning 🤖
Machine learning was implemented in Colab Notebook. Several models have been tried: **Multivariate Linear Regression, Ridge, Lasso, Descision Tree and XGBoost**. 
Their metrics have been analysed and the **importance of coeficients has been compared**.
Based on this analysis, the chosen model was a **Multivariate Linear Regression Model**. This is the model which is used in the API.
The full code is available in this repository.

