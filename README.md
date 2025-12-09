# **Schema Sentinel Code**
This repository contains the code used for our data analytics project. The repository has been organized into three folders: Data Processing, Visualizations, and Algorithms. The files in this repository were used to load the dataset, clean and condition the data, generate the visualizations, and apply analytical algorithms using NYC collisions and weather data.

## **Data Processing**
This folder contains scripts used to load, merge, and prepare the NYC Open Data collision datasets (Crashes, Vehicles, and People) using the shared collision_id key. These datasets provide temporal, geographic, behavioral, and demographic variables, and were limited to the last five years for analysis. Weather data from NOAA were also integrated to add temperature, precipitation, and visibility conditions. The combined data was cleaned, filtered, and transformed to support analysis of collisions in New York City.

## **Visualizations**
This folder includes code used to generate the project’s visualizations, such as:
- Weather Conditions and Collision Risk 
- Contributing Factors and Collision Risk 
- Unlicensed Drivers and Collision Risk 
- Severity by Factor
- Time of Day and Collision Factors
- NYC Most Common Collision Locations
- NYC COllision Trends Over Time
- Injury Rate by Age Gropup and Gender

## **Algorithms**
This folder contains the code used to apply each algorithm and evaluate its contribution to the project’s analysis. The algorithms explored in this project include Multinomial Logistic Regression, Poisson Regression, Time-Series Decomposition, Decision Tree Classification, Random Forest Classification, and XGBoost.

## **Project Repository Link**
This repository is publicly accessible using this link: https://github.com/soohlim96/SchemaSentinelCode
