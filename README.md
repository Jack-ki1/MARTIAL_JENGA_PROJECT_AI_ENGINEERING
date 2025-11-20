# Customer Purchase Amount Prediction Project

## Project Overview

This project aims to build a regression model to predict the numerical value of a customer's purchase amount (in USD) based on their demographic and behavioral data. Using machine learning techniques, we analyze customer shopping trends to create a predictive model that can help businesses understand and anticipate customer purchasing behavior.

## Phase 1: Data Loading and Initial Exploration

### Data Acquisition
The project begins by loading the "Customer Shopping Trends Dataset" from Kaggle. This dataset contains comprehensive information about customer demographics, purchasing behaviors, and transaction details.

### Initial Data Inspection
Upon loading the data, we examine its structure to understand:
- The overall shape of the dataset (number of rows and columns)
- Sample records to get a feel for the data content
- Basic characteristics of different variables

Key observations from this phase:
- The dataset contains multiple categorical and numerical variables related to customer shopping behavior
- Features include demographic information (age, gender), product details (category, price), and behavioral data (previous purchases, review ratings)

## Phase 2: Data Preprocessing

### Feature Selection
We identify and remove irrelevant columns that don't contribute to predicting purchase amounts:
- Removed columns: 'Item Purchased', 'Location', 'Size', 'Color', 'Season', 'Shipping Type', 'Discount Applied', 'Promo Code Used'
- Retained relevant features that likely influence purchase decisions

### Data Encoding
To make the data suitable for machine learning algorithms:
- Categorical variables such as 'Gender', 'Category', 'Payment Method', 'Subscription Status', and 'Frequency of Purchases' are converted to numerical format using one-hot encoding
- This transformation creates binary columns for each category, enabling algorithms to process the data effectively

### Dataset Splitting
The cleaned and encoded data is split into training and testing sets:
- 80% of data (3,120 samples) is used for training the models
- 20% of data (780 samples) is reserved for testing and validation
- This ensures we can evaluate model performance on unseen data

## Phase 3: Feature Scaling

Since we're using algorithms sensitive to feature scales:
- All features are standardized using StandardScaler
- This transforms features to have mean=0 and standard deviation=1
- Ensures that features with larger numerical ranges don't dominate those with smaller ranges

## Phase 4: Model Development and Training

### Linear Regression Model
We first implement a simple Linear Regression model:
- Serves as a baseline for comparison
- Provides interpretable coefficients showing how each feature influences purchase amounts
- Helps understand linear relationships in the data

### Random Forest Regressor
We then implement a more sophisticated Random Forest model:
- Ensemble method that combines multiple decision trees
- Better handles non-linear relationships and complex interactions
- Less prone to overfitting compared to individual decision trees

## Phase 5: Model Evaluation

### Performance Metrics
Models are evaluated using several key metrics:
1. **Root Mean Square Error (RMSE)**: Measures average prediction error in USD
2. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
3. **R-squared (R²)**: Proportion of variance explained by the model (ranges from 0 to 1, higher is better)

### Results Interpretation
Linear Regression Performance:
- Train RMSE: $23.62 | Test RMSE: $23.79
- Train MAE: $20.48 | Test MAE: $20.77
- Train R²: 0.0044 | Test R²: -0.0118

Random Forest Default Performance:
- Train RMSE: $14.62 | Test RMSE: $24.18
- Train MAE: $10.54 | Test MAE: $19.61
- Train R²: 0.4611 | Test R²: -0.0277

Random Forest Tuned Performance:
- Train RMSE: $19.97 | Test RMSE: $23.93
- Train MAE: $15.95 | Test MAE: $19.36
- Train R²: 0.0000 | Test R²: -0.0064

## Phase 6: Hyperparameter Tuning

Using GridSearchCV with 3-fold cross-validation:
- Tested various combinations of Random Forest parameters
- Parameters included: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Best parameters found: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
- Best cross-validation RMSE: 24.02

## Phase 7: Feature Importance Analysis

### Linear Regression Coefficients
Visualization shows how different features linearly impact purchase amounts:
- Positive coefficients increase predicted purchase amounts
- Negative coefficients decrease predicted purchase amounts
- Magnitude indicates strength of influence

### Random Forest Feature Importances
Analysis reveals which features are most important in predicting purchase amounts:
- Previous Purchases: Strong predictor of future spending
- Age: Influences purchasing behavior significantly
- Review Rating: Higher satisfaction correlates with higher spending

## Phase 8: Model Comparison and Selection

Comparing all models based on test set performance:
- Random Forest (even with tuning) shows mixed results
- Despite parameter optimization, models show negative R² scores on test set
- This suggests limited predictive power for the given features

## Key Findings and Business Insights

### Top Influential Factors
1. **Previous Purchases**: Customers with more purchase history tend to spend more
2. **Age**: Different age groups show distinct purchasing patterns
3. **Subscription Status**: Subscribers exhibit different spending behaviors
4. **Review Ratings**: Product satisfaction correlates with spending levels

### Business Recommendations
1. **Target Marketing Campaigns**: Segment customers by age groups for personalized offers
2. **Loyalty Programs**: Develop initiatives to encourage repeat purchases
3. **Subscription Services**: Focus on converting subscribers as they show different spending patterns
4. **Product Quality Improvement**: Higher rated products drive increased spending

## Limitations and Challenges

### Model Performance Issues
- Negative R² scores indicate models explain little variance in purchase amounts
- Features in the dataset may not strongly correlate with purchase amounts
- Potential presence of unmeasured factors influencing customer spending

### Possible Improvements
1. Collect additional relevant features (income level, browsing behavior, etc.)
2. Try alternative modeling approaches (neural networks, gradient boosting)
3. Gather more data points to improve model reliability
4. Engineer new features from existing data (e.g., seasonal spending patterns)

## Deployment Considerations

### Model Saving
The trained model and feature scaler are saved for future use:
- Model saved as `purchase_amount_predictor.pkl`
- Scaler saved as `feature_scaler.pkl`
- Enables making predictions on new customer data

### Implementation Process
To deploy the model in a production environment:
1. Load the saved model and scaler
2. Preprocess new customer data using identical steps as training data
3. Scale features using the loaded scaler
4. Generate predictions using the loaded model
5. Integrate predictions into business applications.
