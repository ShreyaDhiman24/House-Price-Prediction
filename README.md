# Real Estate Price Prediction Website

## Problem Statement:
1. **Uncertainty in Property Prices**: Potential homeowners in Bengaluru face difficulties in determining the right price for houses due to fluctuating market conditions and a lack of trust in property developers.
2. **Market Complexity**: With numerous properties available across different price ranges, buyers need better insights to make informed decisions.

## Solution: Creation of a House Price Prediction Model
1. Data Collection
2. Model Development
3. User-Friendly Interface

## Benefits:
- **Informed Decisions**: Homebuyers can make better choices based on accurate price predictions.
- **Market Transparency**: The model can help build trust in the market by providing objective data-driven insights.
- **Time-Saving**: Buyers can quickly get price estimates without needing to conduct extensive research.

By implementing this house price prediction model, potential homeowners in Bengaluru can navigate the complex real estate market more effectively, making the process of buying a home less daunting.

## Overview
This data science project is a real estate price prediction website. The project consists of three main components:

1. **Model Building**: Built a model using sklearn and linear regression with the Bangalore home prices dataset from Kaggle.com.
2. **Python Flask Server**: Developed a Python Flask server that uses the saved model to serve HTTP requests.
3. **Web Interface**: Created a website using HTML, CSS, and JavaScript, which allows users to enter home square footage, the number of bedrooms, etc. The website calls the Python Flask server to retrieve the predicted price.

## Data Science Concepts Covered
- Data load and cleaning
- Outlier detection and removal
- Feature engineering
- Dimensionality reduction

## Technology and Tools Used and Learned
- **Python**
- **Numpy and Pandas** for data cleaning
- **Matplotlib** for data visualization
- **Sklearn** for model building
- **Jupyter Notebook, Visual Studio Code, and PyCharm** as IDEs
- **Python Flask** for the HTTP server
- **HTML/CSS/JavaScript** for the user interface

[Preview.webm](https://github.com/ShreyaDhiman24/House-Price-Prediction/assets/98320971/e1cb70a2-280a-4487-b725-1864442e4f1e)

## Data Preprocessing Steps for Real Estate Price Prediction

## 1. Initial Dataset Overview
- **Dataset Shape**: The initial dataset contains 13,320 rows and 9 columns.
  ```python
  df1.shape  # Output: (13320, 9)
  ```

- **Columns**: 
  ```python
  df1.columns
  Index(['area_type', 'availability', 'location', 'size', 'society',
         'total_sqft', 'bath', 'balcony', 'price'],
        dtype='object')
  ```

## 2. Drop Unnecessary Features
To build a robust model, we will drop features that are not required:
- **Features to Drop**: `['area_type', 'society', 'balcony', 'availability']`
- **Final Dataset Shape**: 
  ```python
  df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
  df2.shape  # Output: (13320, 5)
  ```

## 3. Data Cleaning
### 3.1 Handle Missing Values
- Identify and handle any NA values in the dataset.

### 3.2 Feature Engineering
#### 3.2.1 Create a BHK (Bedrooms Hall Kitchen) Feature
- Add a new feature to represent the number of BHK units.

#### 3.2.2 Process `total_sqft`
- **Range Handling**: If `total_sqft` is in the format "min-max" (e.g., "2100-2850"), take the average:
  ```python
  total_sqft_avg = (min_value + max_value) / 2
  ```

- **Unit Conversion**: For cases like "34.46 Sq. Meter", convert to square feet using the conversion factor (1 Sq. Meter = 10.764 Sq. Feet).

- **Dropping Corner Cases**: Any ambiguous cases should be dropped to simplify processing.

#### 3.2.3 Create Price per Square Foot Feature
- Add a new feature for price per square foot:
  ```python
  df5 = df4.copy()
  df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
  df5.head()
  ```

## 4. Dimensionality Reduction
### 4.1 Categorical Variable - Location
- Examine the `location` variable, which is categorical.
- Tag any location with less than 10 data points as "other" to reduce categories:
  ```python
  df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
  ```

### 4.2 Unique Locations
- Count unique locations after categorization:
  ```python
  len(df5.location.unique())
  ```

## 5. Outlier Removal
### 5.1 Outlier Removal Using Business Logic
- **Square Footage Logic**: Use business insights to establish that square footage per bedroom should be at least 300 sqft:
  - Example: A 2 BHK apartment should have a minimum of 600 sqft. Remove apartments that do not meet this criteria.

### 5.2 Outlier Removal Using Standard Deviation and Mean
- Identify price per square foot variations:
  - Min price per sqft: 267 Rs/sqft
  - Max price per sqft: 12,000,000 Rs/sqft
- Remove outliers based on mean and standard deviation for each location:
  - Build a dictionary of stats per BHK to identify properties with anomalous pricing.

### 5.3 Consistency in Pricing
- Ensure that for the same location, a 2 BHK apartment is priced higher than a 1 BHK apartment with the same square footage.

### 5.4 Bathroom Count Logic
- Limit the maximum number of bathrooms to the number of bedrooms plus one:
  - Example: In a 4-bedroom home, the maximum bathrooms allowed would be 5.

## 6. One Hot Encoding for Location
- Convert the `location` categorical variable into numerical format using One Hot Encoding to prepare the data for modeling.

## Model Building and Evaluation

## 1. **Data Preparation**

- **Shape of Data**: After preprocessing, our dataset consists of 7239 samples and 244 features (243 input features plus 1 target variable).
- **Target Variable**: The target variable is `price`, which we aim to predict based on various features.

### Rationale:
- **Why Drop 'price'**: We drop the target variable (`price`) from the feature set (`X`) to ensure our model is trained only on the independent variables.

---

## 2. **Train-Test Split**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```
- **Splitting Data**: The dataset is split into training and testing sets, with 80% for training and 20% for testing.

## 3. **Model Training**
```python
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
```
- **Linear Regression Model**: We instantiate and fit a linear regression model using the training data.

### Rationale:
- **Why Linear Regression**: Linear regression is a fundamental algorithm used for predicting continuous outcomes. Given the nature of the target variable (price), which is continuous, this model is suitable.
- **Multiple Parameters**: Although there are multiple parameters (features) involved, linear regression can handle multiple inputs and provide a simple interpretation of how each feature contributes to the prediction.

---

## 4. **Model Evaluation**
```python
lr_clf.score(X_test, y_test)
```
- **Model Accuracy**: The score method returns the coefficient of determination (R²) for the model on the test data.

### Rationale:
- **R² Score Interpretation**: The R² score indicates how well the independent variables explain the variability of the target variable. A score of approximately 0.863 suggests a strong correlation between the features and the target variable.

---

## 5. **K-Fold Cross-Validation**
To further evaluate the model's performance, we can use K-Fold cross-validation:
```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Applying K-Fold cross-validation
cv_scores = cross_val_score(lr_clf, X, y, cv=5)
mean_cv_score = np.mean(cv_scores)
```
- **K-Fold Cross-Validation**: The dataset is split into K subsets (folds). The model is trained on K-1 folds and tested on the remaining fold. This process is repeated K times.

### Rationale:
- **Why Use K-Fold Cross-Validation**: This technique provides a more reliable estimate of the model's performance by using multiple train-test splits. It reduces the variance associated with a single trial of train-test split.

---

## Benefits of This Approach
- **Robustness**: The combination of train-test splitting and K-Fold cross-validation ensures that the model is robust and not overfitting to the training data.
- **Interpretability**: Linear regression allows for easy interpretation of the coefficients, helping to understand the influence of each feature on the target variable.
- **Efficiency**: Linear regression is computationally efficient, making it suitable for large datasets with many features.

---

## Frequently Asked Questions

### Q1: Why not use multiple regression instead of linear regression?
- **A1**: "Multiple regression" generally refers to linear regression with multiple independent variables. The terms can be used interchangeably. In this case, we are using multiple features to predict the target variable.

### Q2: What are the limitations of linear regression?
- **A2**: Linear regression assumes a linear relationship between the independent and dependent variables. It may not perform well if the actual relationship is non-linear.

### Q3: How do you handle non-linear relationships in data?
- **A3**: Non-linear relationships can be addressed by transforming features (e.g., polynomial features) or using more complex models (e.g., decision trees, ensemble methods).

### Q4: What happens if there are multicollinearity issues among features?
- **A4**: Multicollinearity can affect the stability of the coefficient estimates. It can be assessed using Variance Inflation Factor (VIF) and can be addressed by removing or combining correlated features.
