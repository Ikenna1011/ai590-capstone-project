# ai590-capstone-project

# aai590-capstone-project

# Predictive Model for Production Rates - Capstone Project

## Overview

Welcome to the GitHub repository for my capstone project, which focuses on developing a predictive model for production rates using historical data from SMS SITE 002, RIG-2 dataset. The primary objective is to create a reliable model that accurately forecasts future production rates based on temperature, pressure, and flow rate features. This project aims to optimize production planning and detect potential issues or anomalies that could impact the production process.

## Repository Contents

### 1. [Jupyter Notebook](https://github.com/Ikenna1011/ai590-capstone-project/blob/main/IOpurum_AAI_590_Capstone_Project.ipynb)

This Jupyter Notebook provides a step-by-step guide to the entire process, from data processing and exploratory data analysis to model development and evaluation. The notebook is well-documented, making it easy for anyone with no prior knowledge of the project to understand and reproduce the work.

#### Key Sections in the Notebook:

- **Data Processing and Analysis:**
  - Importing the Dataset
  - Exploratory Data Analysis (EDA)
  - Identifying Patterns and Trends
  
- **Data Cleaning:**
  - Handling Missing Values
  - Addressing Outliers
  - Ensuring Data Consistency
  
- **Feature Engineering:**
  - Dimensionality Reduction using Principal Component Analysis (PCA)
  - Creating New Features to Capture Underlying Patterns

- **Model Development:**
  - Building a Deep Neural Network with Keras Sequential API
  - Constructing a Multilayer Perceptron (MLP)
  - Implementing a Random Forest Model
  
- **Model Evaluation:**
  - Comparing Results of Different Models
  - Considering Strengths and Weaknesses
  - Exploring a Voting Ensemble Approach

### 2. [Python Script](https://github.com/Ikenna1011/ai590-capstone-project/blob/main/IOpurum_AAI_590_Capstone_Project.py)

For those who prefer a more straightforward implementation, the Python script contains the exact codes from the Jupyter Notebook. This script allows for easier execution and integration into other projects.

## Project Findings and Insights

Apart from the code files, this repository includes a detailed [project report](https://github.com/Ikenna1011/ai590-capstone-project/blob/main/IOpurum_AAI_590_Capstone_Project_report.pdf) summarizing the findings and insights obtained from the analysis. The report covers:

- **Methodology:**
  - Overview of the Approach Taken
  - Explanation of Data Analysis Techniques Used

- **Results:**
  - Presentation of Model Outputs
  - Interpretation of Model Performance

- **Recommendations:**
  - Suggestions for Improving Production Planning
  - Insights for Enhancing Anomaly Detection

## Machine Learning Models

The process of creating three machine learning models for the project—a deep neural network with Keras sequential API, a multilayer perceptron, and a random forest—is summarized below:

1. **Data Import and Exploratory Data Analysis (EDA):**
   - Understanding the Dataset
   - Identifying Patterns and Trends

2. **Data Cleaning:**
   - Handling Missing Values
   - Addressing Outliers
   - Ensuring Data Consistency

3. **Feature Engineering using PCA:**
   - Dimensionality Reduction
   - Creating New Features

4. **Model Building:**
   - Random Forest Algorithm:
     - Handling Complex Datasets
     - Identifying Non-linear Relationships
   - Multilayer Perceptron (MLP) Algorithm:
     - Handling Large Datasets
     - Identifying Complex Patterns
   - Deep Neural Network:
     - Versatility in Handling Complex Datasets

5. **Model Comparison:**
   - Evaluating Results of Each Model
   - Considering Strengths and Weaknesses

6. **Voting Ensemble Approach:**
   - Utilizing a Voting System with All Three Models
   - Making Informed Predictions for Gas, Oil, and Water Rates

## Important Note

While machine learning models and data analysis techniques have the potential to enhance operational efficiency and productivity, it is crucial to emphasize that the success of these models heavily relies on the quality and accuracy of the data being used. Therefore, it is recommended to ensure the dataset is well-prepared and validated before applying the models to real-world scenarios.

Feel free to explore the code files, report, and dive into the details of the project. If you have any questions or suggestions, please don't hesitate to reach out. Happy exploring!
