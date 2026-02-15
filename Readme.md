ML Assignment 2: Classification and Deployment

a. Problem Statement
The goal of this project is to predict whether a bank client will subscribe to a term deposit based on a variety of demographic and marketing features from the Bank Marketing dataset.

b. Dataset Description
Source: Kaggle (Bank Marketing Dataset)
Instances: 41,188
Features: 20
Target Variable: 'y' (Binary classification: "yes" or "no").

c. Models Used
The following six models were implemented and evaluated on the same dataset:

Comparison Table 

------------------------------------------------------------------------------------------------
|ML Model Name        |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC |
|:--------------------|-----------:|------------:|------------:|---------:|-----------:|------:|
| Logistic Regression |      0.911 |       0.933 |       0.672 |    0.42  |      0.517 | 0.487 |
| Decision Tree       |      0.887 |       0.732 |       0.502 |    0.532 |      0.516 | 0.452 |
| KNN                 |      0.902 |       0.87  |       0.586 |    0.468 |      0.52  | 0.47  |
| Naive Bayes         |      0.865 |       0.829 |       0.422 |    0.521 |      0.466 | 0.392 |
| Random Forest       |      0.91  |       0.938 |       0.652 |    0.448 |      0.531 | 0.494 |
| XGBoost             |      0.915 |       0.945 |       0.646 |    0.548 |      0.593 | 0.548 |
------------------------------------------------------------------------------------------------


Model Performance Observations 

---------------------------------------------------------------
| ML Model Name       |   Observations                        | 
|:--------------------|--------------------------------------:|
| Logistic Regression |  High accuracy but lower recall.      | 
| Decision Tree       |  Faster training but lower AUC        |  
| KNN                 |  Balanced but sensitive to neighbors. | 
| Naive Bayes         |  Lowest performer for this dataset.   |  
| Random Forest       |  Strong robustness and AUC.           | 
| XGBoost             |  Best overall performer for this task.|
---------------------------------------------------------------