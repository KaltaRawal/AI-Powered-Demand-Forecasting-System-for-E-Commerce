# AI-Powered-Demand-Forecasting-System-for-E-Commerce
## Project Overview
This repository holds a robust Time-Series Forecasting solution that is designed to generate accurate predictions of both total daily sales and daily demand across multiple product categories in an e-commerce platform. The project applies advanced machine learning and statistical techniques to provide actionable insights that are critical to inventory optimization and operational efficiency improvement in the highly competitive landscape of e-commerce.
## Problem Statement
Accurate time-series forecasting is crucial for optimizing inventory and supply chain management, reducing costs, and improving customer satisfaction in the highly competitive e-commerce industry. This project aims to:

* Predict daily sales for the e-commerce platform.
* Forecast daily demand across multiple product categories.
### Key Business Metrics
* Inventory Turnover Rate: Measures how quickly inventory is sold and replaced.
* Customer Satisfaction Score (CSAT): Ensures product availability for timely deliveries.
* Operational Costs: Helps control costs related to storage and shipping.
* Stockout Rate: Reduces the frequency of stockouts, ensuring products are available when needed.
### Objectives and Core Functionality
The central goal of this project is to address the challenge of accurate time-series forecasting within a dynamic retail environment.
* Total Daily Sales Prediction - Forecast the aggregate sales value for the entire e-commerce platform.
* Multi-Category Demand Forecasting - Predict the daily demand (order count) for individual products grouped into 56 distinct product categories
### Business Benefits
* Inventory Optimization - Reduces risk of over-stocking or under-stocking to maintain optimal stock levels.
* Operational Efficiency - Reduces storage, shipping, and handling costs by aligning inventory levels with forecasted demand.
* Customer Satisfaction - Lowers the Stockout Rate and ensures product availability for timely deliveries.
* Business Intelligence	- Provides real-time forecasts to evaluate marketing effectiveness and inform strategic business decision-making.
## Dataset 
The dataset represents a set of orders done through Olist Store, one of the large e
commerce platforms in Brazil. This is a normalized dataset with several interlinked 
tables describing customers, orders, payments, products, sellers, and reviews. 
## Features and Methodology
The solution employs a dual-modeling framework, contrasting a classic statistical baseline with a highly effective **Machine Learning** approach, as detailed in the system architecture diagram.

### 1. Data Processing and Feature Engineering
[cite_start]The reliability of the forecast hinges on high-quality input data and **advanced feature engineering**[cite: 11].

* **Source Data:** Brazilian E-commerce Public Dataset by Olist (Kaggle).
* [cite_start]**Preprocessing:** Orders were filtered for `delivered` status [cite: 12][cite_start], and multiple transactional tables were **merged** into a **unified time-series format**[cite: 13].
* **Feature Engineering:**
    * [cite_start]**Temporal Features:** Extracted `day_of_week` and `month`[cite: 14].
    * [cite_start]**Autoregressive (AR):** Incorporated **lag features** (e.g., sales from prior days)[cite: 16].
    * [cite_start]**Moving Average (MA):** Calculated **rolling statistics** (rolling means)[cite: 16].
    * [cite_start]**Exogenous Variables:** Added **holiday indicators** to capture external demand spikes (e.g., **Black Friday**)[cite: 15].

### 2. Forecasting Models and Architecture

[cite_start]The architecture utilizes a comparative setup to validate the superiority of the **XGBoost** Regressor[cite: 17].

| Forecasting Model | Focus | Key Technique |
| :--- | :--- | :--- |
| **SARIMAX** | Statistical Baseline | [cite_start]Accounts for **seasonality** and **trend** with **exogenous variables**[cite: 18]. |
| **XGBoost** | **Machine Learning** | [cite_start]Designed to **capture complex nonlinear dependencies** using engineered features[cite: 19]. |
| **Global XGBoost Model** | **Scalability** | [cite_start]Implemented **Multi-Output Regression** to predict demand across **56 product categories** concurrently[cite: 20]. |
