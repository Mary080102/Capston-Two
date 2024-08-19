# Predicting Asthma Hospital visits based on Air Pollution Levels
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://u4d2z7k9.rocketcdn.me/wp-content/uploads/2022/11/Untitled-1024-×-683px-17.jpg.webp">
 <source media="(prefers-color-scheme: light)" srcset="https://u4d2z7k9.rocketcdn.me/wp-content/uploads/2022/11/Untitled-1024-×-683px-17.jpg.webp">
 <img alt="YOUR-ALT-TEXT" src="https://u4d2z7k9.rocketcdn.me/wp-content/uploads/2022/11/Untitled-1024-×-683px-17.jpg.webp">
</picture>

## 1.Introduction
The core issue addressed in this study is the relationship between air quality and respiratory problems, specifically asthma-related hospital visits, in New York City. This project aims to determine whether atmospheric pollutants significantly impact the frequency of asthma-related hospital visits among NYC's residents.
New York City, being a densely populated urban area, faces notable challenges related to air pollution. This study is crucial given the increasing concerns about environmental health, urban living, and sustainable city planning. The findings will have significant implications for policy decisions, public health initiatives, and raising individual awareness about the health impacts of air quality.

## 2.Data
The data utilized in this study is sourced from the NYC Environment & Health Data Portal (NYC.gov), which provides comprehensive information on environmental pollutants and their health impacts. The dataset was downloaded from the NYC.gov website.To view the original data click on the links below:

* [Air quality](https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/air-quality/?id=2023#display=summary)
* [Asthma](a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/asthma/?id=2414#display=summary)

             

## 3.Data Wrangling

[Data Cleaning Notebook](https://github.com/Mary080102/Capston-Two/blob/202f588a71fc53fcdf9a611ef8f2c41660066847/notebooks/Air%20quality%20Capston-%20Data%20wrangling.ipynb)

The raw dataset from NYC Environment & Health contains 18 columns and 2,037,616 rows, requiring significant size reduction.

* **Problem1:** Some columns in the asthma dataset that were based on an unclear counting method. I Dropped unclear columns in the asthma dataset (e.g., "Estimated annual rate per 10,000") and in the PM2.5 dataset (e.g., "10th percentile mcg/m3" and "90th percentile mcg/m3").
  
* **Problem 2:** I noticed time period discrepancies between the asthma and air quality data .Chose to use yearly records for consistency and discarded seasonal data.
  
* **Problem 3:** No null values, but data types of some columns needed changes and prefixes removal.
  
* **Problem 4:** Dropped unnecessary geographical columns that were repetitive or lacked useful information.

## 4.EDA

[EDA Notebook](https://github.com/Mary080102/Capston-Two/blob/202f588a71fc53fcdf9a611ef8f2c41660066847/notebooks/Air%20quality-%20Exploratory%20Data%20Analysis.ipynb)

Pairwise relationships in the dataset were examined to understand the interactions and correlations between different variables.

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot1.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot1.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot1.png">
</picture>

Correlation Matrix to look at the relationships between variables in our datasets.

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot2.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot2.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/PNG/plot2.png">
</picture>

## 5.Pre-Processing
[Pre-Processing Notebook](https://github.com/Mary080102/Capston-Two/blob/b5869fbe7a5d5be1db3f6e10ba9088ad16314487/notebooks/Pre-processing%20and%20Training%20Data%20Development.ipynb)

In the pre-processing step, we transformed the categorical feature 'GeoType' as it was necessary for the machine learning models, while other categorical features were excluded. Scaling was applied to models such as Linear Regression, Ridge Regression, Gradient Boosting, and SVR, but not to the Random Forest model due to its insensitivity to feature scale. Evaluation based on RMSE scores indicated that scaling improved performance, with normalized data outperforming standardized data, leading us to choose normalization as the preferred scaling method.

## 6.Algorithms & Machine Learning
[Modelin Notebook]()

This is a regression problem, in supervised learning. Here we have used the following regression models:
* Multiple Linear Regression
* Ridge Regression (Regularized Linear Regression)
* Gradient Boosting Machines
* Random Forest
* Support Vector Regression Model(SVR)

I applied different ML models above and evaluated their performances using cross-validation for both the training and test data. 

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot3.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot3.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot3.png">
</picture>


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot4.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot4.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot4.png">
</picture>


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot5.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot5.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot5.png">
</picture>

Based on the above metrics results, **Random Forest** and **Gradian Boosting** are the best models among the ones compared. They have the highest R². However, the high values of MSE and MAE suggest that the actual predictions may still be quite far from the actual values. To make the process more accurate I want to use feature selection in the next step.


I applied hyperparameter tuning for both models selected. Results of the performance metrics for two machine learning models (below table)shows both models have high Best Score and Test Score values, indicating strong performance in predicting asthma-related hospital visits. The Random Forest model slightly outperforms the Gradient Boosting model, as indicated by the marginally higher Best Score and Test Score values. The small difference between Best Score and Test Score for both models suggests that they generalize well and are not overfitting to the training data.



| Model        | Best Score     | Test score    |R2    |
| :---:         |     :---:     |       :---:   | :---:|
| Random Fores   | 0.945868     | 0.954561    |     0.954561 |
| Gradient Boosting     | 0.950065       | 0.944047      |   0.944047   |


### 6.1 Analysis of Residual Plot

The below plots shown the residuals (the differences between the actual and predicted values) are centered around zero, which is a good sign. This indicates that the model is generally unbiased. However, there is a spread of residuals, with some large positive and negative residuals. This could suggest that the model might be underestimating or overestimating some points.
The residuals seem to be fairly evenly distributed, with no clear pattern or trend. This is desirable, as it indicates that the model's errors are random rather than systematic. There are a few points with very high positive and negative residuals, which can be considered outliers. These are data points where the model's predictions are significantly different from the actual values. These outliers can impact the model's performance metrics and may warrant further investigation to understand their nature (e.g., data entry errors, special cases, etc.).


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot6.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot6.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot6.png">
</picture>


<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot7.png">
 <source media="(prefers-color-scheme: light)" srcset="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot7.png">
 <img alt="YOUR-ALT-TEXT" src="https://github.com/Mary080102/Capston-Two/blob/7d04797565f336ea74700806274c9eeceaf97eb7/PNG/plot7.png">
</picture>

### 8.Conclusion
In this modeling project, multiple machine learning algorithms were applied to predict asthma-related hospital visits in New York City based on various air quality and geographical features. The models evaluated include Linear Regression, Ridge Regression, Gradient Boosting, Random Forest, and Support Vector Regression (SVR).
The performance metrics (R2, MAE, and MSE) showed that both Gradient Boosting and Random Forest performed exceptionally well, with R2 values of 0.962973 and 0.960578, respectively. These models also demonstrated lower MAE and MSE values compared to the other models, indicating better prediction accuracy and precision.
* The feature importance plots revealed that geographical features, particularly GeoType_Citywide, had the most significant impact on the predictions, followed by air quality metrics such as pm/Mean mcg/m3 and no2/Mean ppb.

* Hyperparameter tuning using Grid Search CV was performed to optimize the models. For Random Forest, the best parameters included max_depth: 10, max_features: 'sqrt', and n_estimators: 200. For Gradient Boosting, the optimal parameters included learning_rate: 0.1, max_depth: 5, and n_estimators: 100.

* Residual plots indicated that while the models performed well, there are still some outliers.


### 9. Future Overseeing

* Further explore and engineer features that could improve model performance, such as incorporating additional air quality metrics or socioeconomic factors.

* Collect more data, especially from air pollutants, different time periods or additional geographical areas, to improve model generalization.
  
* Explore advanced algorithms such as XGBoost, LightGBM, or neural networks to potentially capture complex relationships within the data.


* Investigating the temporal dynamics of air quality and asthma exacerbations by incorporating time-series analysis could reveal seasonal or temporal trends that are not captured by static models. This could help in understanding how different times of the year or specific weather conditions affect asthma incidence.
  
* Conducting more detailed geospatial analysis using advanced GIS tools could help in identifying specific areas within the city that are more prone to poor air quality and higher asthma rates. This could inform targeted interventions and policy decisions.
