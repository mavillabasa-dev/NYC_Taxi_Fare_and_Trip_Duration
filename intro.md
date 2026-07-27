# NYC Taxi Fare and Trip Duration Prediction

## Executive Summary
Industries: Transportation, Ridesharing, Logistics.

Technologies and Tools: Supervised Learning, Neural Networks, Time-series analysis, Scikit-learn, Pandas, TensorFlow, HTTP APIs, Docker.

The availability of millions of ride data in New York City provides a unique opportunity to gain insight into the patterns of city life, including traffic flow, road closures, and large-scale events. Ridesharing apps have gained popularity over the years, and taxi companies are under increasing pressure to provide accurate fare and duration estimates to their customers in order to stay competitive. 

This project aims to predict the fare and duration of a taxi ride in New York City using data that would be available at the beginning of the ride, such as pickup and dropoff coordinates, trip distance, start time, number of passengers, and a rate code detailing whether the standard rate or the airport rate was applied. 

The predicted fare and duration can help passengers make informed decisions about when to start their commute and help drivers decide which of two potential rides will be more profitable. Furthermore, providing visibility into fares can attract customers during peak hours when ridesharing services implement surge pricing.

To achieve this, the project will involve building predictive models for taxi ride fare and duration. The models will be trained on a dataset of New York City taxi rides, which will be split into training and testing sets. The model's performance will be evaluated based on its ability to predict the fare and duration of a taxi ride accurately.

In addition to predicting the fare and duration of a taxi ride, this project can also be extended to analyze the factors that impact the fare and duration of a ride. Factors such as traffic flow, weather, and road closures. Such insights can help taxi companies optimize their operations and improve customer satisfaction.

On the other hand, we can also use the provided dataset to predict Taxi demand, this is, predict the number of pickups as accurately as possible in any region and nearby regions. Such prediction will help dispatchers immensely in making important decisions that could revive their profit margin.

Overall, this project provides an opportunity to apply machine learning techniques to real-world data and gain insights into the transportation industry in New York City.

## Overview
The main objective of this project is to build a machine learning model that can predict the fare and duration of taxi rides in New York City. The model will take into account various factors such as pick-up and drop-off locations, date and time of the ride, and weather conditions. This project is similar to what you have implemented in the last few sprints but with a focus on building a model that can handle time-series data.

## Deliverables
Goal: The main goal of this project is to predict the fare and duration of taxi rides from the given dataset. You can use two different models or a single model with two outputs. We suggest you start with some simple model as a baseline (Linear Regression or Decision Trees) and then move to more complex algorithms like XGBoost or Neural Networks.

In order to graduate from the ML Developer Career, you have to approve the Main Deliverables. You are also welcome to complete the Optional Deliverables if you want to continue to add experience and build your portfolio, although those are not mandatory.

### Main Deliverables:

1. Exploratory Dataset Analysis (EDA) Jupyter notebooks and dataset
2. Scripts used for data pre-processing and data preparation
3. Training scripts and trained models. Description of how to reproduce results
4. Implementation and training of model for fare and duration prediction, because of the size of the dataset, you can a single month of data to train and evaluate your model (e.g. February 2022)
5. Present results and a demo of the model doing predictions in real-time preferably using an API
Everything must be containerized using Docker

> [!NOTE] 
> we will provide reference literature for you to better understand this project, but you will be able to solve this problem without implementing the architecture presented in the paper.

### Additional Optional Deliverables:

1. Factors such as traffic flow, weather, and road closures can be analyzed to gain insights into how they affect the fare and duration of a taxi ride. You can extend the current dataset features using third-party services to get this data. For example, you can use OpenWeather API to retrieve historical wheater data and add that to your dataset.
2. Train a new model for Taxi demand prediction for a given region based on date and time.

## Approach and Milestones
There are many ways to approach this project and at first sight, this might seem very overwhelming. A good rule of thumb is this:

Get a good overview and idea of what you need to build (craft an architecture diagram with your team).
Identify the unknowns and, most importantly, the major risks for the project and allocate time appropriately. For instance: setting up a Dockerized API is simple and low risk but dealing with a dirty dataset is a high-risk task. Present your plan to your mentor to get his insights.
Some tasks will take a long time and might be a blocker for other tasks. Identify those risky tasks first and try to divide and conquer as much as possible.
One possible approach is this milestone/project plan:

| Milestone | Description |
|-----------|-------------|
| Setup repository and project structure | Create the GitHub repository. Organize the project, create subfolders, and prepare/mock as much of the project structure as necessary for the final deliverables. |
| State-of-the-art review | Read and understand the main concepts behind the referenced papers. You do not need to master the specific architectures, but this review will provide a better understanding of the problem domain and how others have approached and solved it. |
| Download and evaluate the dataset | Download the dataset and perform an Exploratory Data Analysis (EDA). Collect metrics such as the number of samples, features, missing values, outliers, correlations, and other relevant statistics to understand the dataset structure. |
| Create a training dataset | Clean the data, create a database, and store the processed dataset for training. |
| Taxi Fare and Duration models training | Train and evaluate multiple regression models, including Linear Regression, Decision Trees, LightGBM, XGBoost, Random Forest, Ensemble methods, stacked variants, and a Multi-Layer Perceptron (MLP). |
| Evaluate/test the initial classifier | Compare model performance using metrics such as MAE, MSE, training time, and inference time. Select the best-performing model based on predictive accuracy and inference efficiency. |
| Present results and demo them | Prepare a clean and well-structured Jupyter Notebook that loads the best model, runs predictions on sample data, and includes visualizations such as an NYC map and prediction results by region. As an extension, consider deploying the model using Flask or FastAPI to exceed project expectations. |
| Add tests to the main components (Optional) | Implement tests for the main project components to improve reliability and maintainability. |
| Preview service to other teams | Demonstrate the project to other teams, gather feedback, and incorporate final improvements. |
| Build final presentation and prep for demo | Prepare the final presentation and ensure the project is ready for the Demo Day presentation. |


## Dataset

How to access the dataset?

The dataset can be accessed from the Official NYC web site. For this particular project we advise you to use the 2022 TLC Trip Record Data given is the most complete and up-to-date dataset on the site at the moment.

We are going to use only the data corresponding to "Yellow Taxi Trip Records".

Because of the dataset size, it was split into separate files, one for each month of the year. You can start doing experiments using only one month of data. We advise you to use Yellow Taxi Trip Records (PARQUET) - May 2022 as a start.

## References
For this project, we recommend you read the following materials:

Papers and articles

Fare and Duration Prediction: A Study of New York City Taxi Rides
Towards Data Science - NYC Taxi Fare Prediction
New York Yellow Taxi Demand prediction using Machine Learning (Optional part)
Data Dictionaries and MetaData

Trip Record User Guide
Yellow Trips Data Dictionary
Taxi Zone Maps and Lookup Tables

Taxi Zone Lookup Table (CSV)
Taxi Zone Shapefile (PARQUET)