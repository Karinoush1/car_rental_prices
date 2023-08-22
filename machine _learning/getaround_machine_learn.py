# -*- coding: utf-8 -*-
"""depl_ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RZP4aW_1yGU8fShxnReWCic_UKFGXNXU
"""

# Import useful modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # to avoid deprecation warnings

from google.colab import drive
drive.mount('/content/drive')

df_price = pd.read_csv('/content/drive/MyDrive/JEDHA/CERTIF/C_Depl/get_around_pricing_project.csv')
df_price.head(5)

df_price

df_price.iloc[3, :]

df_price.iloc[1, :]

"""{
  "car_model_name": "Citroën",
  "mileage": 13929,
  "engine_power": 150,
  "fuel": "petrol",
  "car_type": "convertible",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": true,
  "winter_tires": true
}
"""

print("Basic Statistics:")
print(df_price.describe())
print('----------------')
print("Data Types and Non-null Counts:")
print(df_price.info())
print('----------------')
print("Number of Rows and Columns:")
print(df_price.shape)
print('----------------')

# Calculate the percentage of missing values in each column
missing_percent = (df_price.isnull().sum() / len(df_price)) * 100
missing_df_price = pd.DataFrame({'Column Name': df_price.columns, 'Missing Percentage': missing_percent})
missing_df_price = missing_df_price.reset_index(drop=True)
missing_df_price = missing_df_price.sort_values('Missing Percentage', ascending=False)
missing_df_price

df_price.columns

df_price['car_model_name']= df_price['model_key']
df_price.drop(columns = ['model_key'], inplace = True )

# Distribution of numeric variables
numeric_features = ['mileage', 'engine_power']
fig = make_subplots(rows = len(numeric_features), cols = 1, subplot_titles = numeric_features)
for i in range(len(numeric_features)):
    fig.add_trace(
        go.Histogram(
            x = df_price[numeric_features[i]], nbinsx = 100),
        row = i + 1,
        col = 1)
fig.update_layout(
        title = go.layout.Title(text = "Distribution of quantitative variables", x = 0.5), showlegend = False,
            autosize=False, height=1000)
fig.show()

# Barplot of qualitative variables
categorical_features = ['car_model_name', 'fuel',
       'paint_color', 'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires']
fig = make_subplots(rows = len(categorical_features), cols = 1, subplot_titles = categorical_features)
for i in range(len(categorical_features)):
    x_coords = df_price[categorical_features[i]].value_counts().index.tolist()
    y_coords = df_price[categorical_features[i]].value_counts().tolist()
    fig.add_trace(go.Bar(x = x_coords,y = y_coords),row = i + 1,col = 1)

fig.update_layout(title = go.layout.Title(text = "Barplot of qualitative variables", x = 0.5), showlegend = False, autosize=False, height=2000)

# Correlation between features : heatmap.
subset = df_price[['car_model_name', 'mileage', 'engine_power', 'fuel',
       'paint_color', 'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires']]

# Compute the correlation matrix
correlation_matrix = subset.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

"""## 🐼🐼 Preprocessing - Pandas.
*   drop columns : useless ('Unnamed', ' paint_color')
*   drop columns : having to many missing values (we do not have missing values)
*   drop rows : outilers (for our projetct, +- 3 standard deviation)
*   drop rows : missing values in target variable (don't have)
*   Convert boolean values in numeric representation
*  Target variable/target (Y) that we will try to predict, to separate from
"""

df_price.drop(columns=['Unnamed: 0', 'paint_color'], inplace=True)

df_price.info()

# convert booleans into integers
df_price.replace({False: 0, True: 1}, inplace=True)
df_price.head(5)

#DROP OUTLIERS
columns_to_check = ['mileage', 'engine_power']

# Calculate the lower and upper bounds for outliers
lower_val = df_price[columns_to_check].mean() - 3 * df_price[columns_to_check].std()
upper_val = df_price[columns_to_check].mean() + 3 * df_price[columns_to_check].std()


# Create a boolean mask for outlier rows
outlier_mask = ~((df_price[columns_to_check] < lower_val) | (df_price[columns_to_check] > upper_val)).any(axis=1)

# Filter the df to include rows without outliers
df_filtered = df_price[outlier_mask]

print(df_filtered)
print(f'There are {len(df_filtered)} rows that do not have outliers.')
print(f'There are {len(df_price)- len(df_filtered)} rows to drop, which represents {round(100-((len(df_filtered)*100 )/len(df_price)), 2)} % of the original dataset. ')

# Separate the target value "rental_price_per_day" from other features
target_name = 'rental_price_per_day'

print("Separating labels from features...")
Y = df_filtered.loc[:,target_name]
X = df_filtered.drop(target_name, axis = 1)
print("...Done.")
print(Y.head())
print()
print(X.head())
print()

# Perform a train / test split
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("...Done.")
print()
# Check the shapes of X_train and X_test
print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

"""##🔬🔬 Preprocessing - Scikit-Learn

*   Impute missing values and remplace with mean, median, most frequent value (categorical values)
*   Encode categorical values with OHE
*   Standardize quantitative values

"""

# Create pipeline for numeric and categorical features
numeric_features = ['mileage', 'engine_power']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['car_model_name','fuel',
        'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('numeric', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Preprocessings on train set
print("Performing preprocessings on train set...")
print(X_train.head())
X_train = preprocessor.fit_transform(X_train)

print('...Done.')
print(X_train[0:5, :])
print()

# Preprocessings on test set
print("Performing preprocessings on test set...")
print(X_test.head())
X_test = preprocessor.transform(X_test)
print('...Done.')
print(X_test[0:5, :])
print()

# Check the shapes of X_train and X_test to ensure that preprocessing has be done correctly
print("The shape of X_train is:", X_train.shape)
print("The shape of X_test is:", X_test.shape)

"""## Linear regression model"""

# Train model
model_lr = LinearRegression()
print("Training model...")
model_lr.fit(X_train, Y_train)
print("...Done.")

# Predictions on training set
print("Predictions on training set...")
Y_train_pred = model_lr.predict(X_train)
print("...Done.")
print(Y_train_pred[0:10])
print()

# Predictions on test set
print("Predictions on test set...")
Y_test_pred = model_lr.predict(X_test)
print("...Done.")
print(Y_test_pred[0:10])
print()

print(" The X_train shape is :", X_train.shape)

#METRICS MSE
# returns : an array of floating point values, one for each individual target.
#that measures the average squared difference between the predicted values and the actual values.
print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", r2_score(Y_train, Y_train_pred))
print("R2 score on test set : ", r2_score(Y_test, Y_test_pred))

# Visualize basline model's metrics
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot for the training set
axs[0].scatter(Y_train, Y_train_pred)
axs[0].plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='red', linestyle='--')
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Base model TRAIN: Actual vs. Predicted Values')

# Scatter plot for the test set
axs[1].scatter(Y_test, Y_test_pred)
axs[1].plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Predicted Values')
axs[1].set_title('Base model TEST: Actual vs. Predicted Values')

plt.tight_layout()
plt.show()

### coef_ attribute returns an array of coefficients, where each coefficient corresponds to a feature in the input data
#print(model_lr.coef_)
#print(' There are {} coefficients.'.format(len(model_lr.coef_)))

#retrieve feature names from preprocessor by using .transformers_
#print(preprocessor.transformers_)

# Access the feature names from the preprocessor
feature_names = preprocessor.transformers_[0][2] + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features))

# Access the coefficients and  the intercept
coefficients = model_lr.coef_
intercept = model_lr.intercept_

# Print the coefficients with corresponding feature names
for feature, coefficient in zip(feature_names, coefficients):
    print(f'{feature}: {coefficient}')

# Print the intercept (if applicable)
print(f'Intercept: {intercept}')

# Create a dataframe to visualize
model_coefficients = pd.DataFrame(index = feature_names, columns = ['LinearRegression'], data = model_lr.coef_)
feature_importance =model_coefficients.sort_values(by = 'LinearRegression', key = np.abs, ascending = False)
feature_importance

"""The Intercept is the base value of the predicted target variable when all the feature values are zero.

"""

# Plot coefficients
fig = px.bar(feature_importance, orientation='h')
fig.update_layout(
    showlegend=False,
    height=1000,
    margin={'l': 200},
    xaxis_tickangle=-45,
    xaxis=dict(title='Coefficient'),
    yaxis=dict(title='Feature', title_standoff=30)
)
fig.show()

"""*   Nobody wants honda or mazda.
*   People value fuel hybrid petrol or electro.
*   Automatic, air conditioning and parking does not have so much influence as expected.
*  GPS and getaround connect are important but not crutial.

# Ridge
"""

# Perform grid search
print("Grid search...")
model_ridge = Ridge()
# Grid of values to be tested
params = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5]
}
best_ridge = GridSearchCV(model_ridge, param_grid = params, cv = 5)
best_ridge.fit(X_train, Y_train)
print("...Done.")
print("Best hyperparameters : ", best_ridge.best_params_)
print("Best R2 score : ", best_ridge.best_score_)

model_ridge = Ridge(alpha = 0.5)
model_ridge.fit(X_train, Y_train)
# Predictions on training set and on test set
Y_train_pred = model_ridge.predict(X_train)
Y_test_pred = model_ridge.predict(X_test)
print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", model_ridge.score(X_train, Y_train))
print("R2 score on test set : ", model_ridge.score(X_test, Y_test))

"""# Lasso"""

# Perform grid search
print("Grid search...")
model_lasso = Lasso()
# Grid of values to be tested
params = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1, 2]
}
best_lasso = GridSearchCV(model_lasso, param_grid = params, cv = 5) # cv : the number of folds to be used for CV
best_lasso.fit(X_train, Y_train)
print("...Done.")
print("Best hyperparameters : ", best_lasso.best_params_)
print("Best R2 score : ", best_lasso.best_score_)

model_lasso_1 = Lasso(alpha = 0.01)
model_lasso_1.fit(X_train, Y_train)
Y_train_pred =  model_lasso_1.predict(X_train)
Y_test_pred =  model_lasso_1.predict(X_test)
print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", model_lasso_1.score(X_train, Y_train))
print("R2 score on test set : ", model_lasso_1.score(X_test, Y_test))

"""## Descision Tree"""

# Decision Tree model
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, Y_train)

# Predict and evaluate
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(Y_test, y_pred_tree)
print(f'Decision Tree MSE: {mse_tree}')
# Print R^2 scores
print("R2 score on training set : ", tree_model.score(X_train, Y_train))
print("R2 score on test set : ", tree_model.score(X_test, Y_test))

"""# Random Forest"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Perform grid search
print("Grid search...")
random_forest = RandomForestRegressor()

# Grid of values to be tested
params = {
    'max_depth': [8,12, 16],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [4, 8, 10],
    'n_estimators': [40, 60, 80]
}
print(params)
gridsearch_rf = GridSearchCV(random_forest, param_grid = params, cv = 3, verbose = 1) # cv : the number of folds to be used for CV
gridsearch_rf.fit(X_train, Y_train)
print("...Done.")
print("Best hyperparameters : ", gridsearch_rf.best_params_)
print("Best validation accuracy : ", gridsearch_rf.best_score_)
print()
print("R2 score on training set : ", gridsearch_rf.score(X_train, Y_train))
print("R2 score on test set : ", gridsearch_rf.score(X_test, Y_test))

# Predictions on training set
print("Predictions on training set...")
Y_train_pred = gridsearch_rf.predict(X_train)
print("...Done.")
print(Y_train_pred[0:10])
print()

# Predictions on test set
print("Predictions on test set...")
Y_test_pred = gridsearch_rf.predict(X_test)
print("...Done.")
print(Y_test_pred[0:10])
print()

print(" The X_train shape is :", X_train.shape)

print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", r2_score(Y_train, Y_train_pred))
print("R2 score on test set : ", r2_score(Y_test, Y_test_pred))

# Get the feature importances
importances = gridsearch_rf.best_estimator_.feature_importances_
importances

# Access the feature names from the preprocessor
feature_names = preprocessor.transformers_[0][2] + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features))

# Access the coefficients
coefficients = gridsearch_rf.best_estimator_.feature_importances_

# Print the coefficients with corresponding feature names
for feature, coefficient in zip(feature_names, coefficients):
    print(f'{feature}: {coefficient}')

model_coefficients = pd.DataFrame(index = feature_names, columns = ['RandomForest'], data = gridsearch_rf.best_estimator_.feature_importances_)
feature_importance =model_coefficients.sort_values(by = 'RandomForest', key = np.abs, ascending = False)
feature_importance

# Visualize basline model's metrics
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot for the training set
axs[0].scatter(Y_train, Y_train_pred)
axs[0].plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='red', linestyle='--')
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Base model TRAIN: Actual vs. Predicted Values')

# Scatter plot for the test set
axs[1].scatter(Y_test, Y_test_pred)
axs[1].plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Predicted Values')
axs[1].set_title('Base model TEST: Actual vs. Predicted Values')

plt.tight_layout()
plt.show()

"""By analizing feature importance, we see, that the model has realy taken in account only numeric features. Numeric features represent only 15.4% of features, which is not enough for our case.

## XGBoost
"""

print("Grid search...")

xgboost = XGBRegressor()
# Grid of values to be tested
params = {
    'max_depth': [6, 8, 10], # exactly the same role as in scikit-learn
    'min_child_weight': [2, 3, 5], # effect is more or less similar to min_samples_leaf and min_samples_split
    'n_estimators': [ 16, 17, 18] # exactly the same role as in scikit-learn
}
print(params)
gridsearch_xgb = GridSearchCV(xgboost, param_grid = params, cv = 3) # cv : the number of folds to be used for CV
gridsearch_xgb.fit(X_train, Y_train)
print("...Done.")
print("Best hyperparameters : ", gridsearch_xgb.best_params_)
print("Best validation accuracy : ", gridsearch_xgb.best_score_)
print()
print("Accuracy on training set : ", gridsearch_xgb.score(X_train, Y_train))
print("Accuracy on test set : ", gridsearch_xgb.score(X_test, Y_test))

# Predictions on TEST set
Y_test_pred = gridsearch_xgb.predict(X_test)
# Predictions on TRAIN set
Y_train_pred = gridsearch_xgb.predict(X_train)

print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", r2_score(Y_train, Y_train_pred))
print("R2 score on test set : ", r2_score(Y_test, Y_test_pred))

# Get the feature importances
feature_names = preprocessor.transformers_[0][2] + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features))
# Access the coefficients
coefficients = gridsearch_xgb.best_estimator_.feature_importances_

# Print the coefficients with corresponding feature names
for feature, coefficient in zip(feature_names, coefficients):
    print(f'{feature}: {coefficient}')

model_coefficients = pd.DataFrame(index = feature_names, columns = ['XGBoost'], data = gridsearch_xgb.best_estimator_.feature_importances_)
feature_importance =model_coefficients.sort_values(by = 'XGBoost', key = np.abs, ascending = False)
feature_importance

# Visualize basline model's metrics
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot for the training set
axs[0].scatter(Y_train, Y_train_pred)
axs[0].plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='red', linestyle='--')
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Base model TRAIN: Actual vs. Predicted Values')

# Scatter plot for the test set
axs[1].scatter(Y_test, Y_test_pred)
axs[1].plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Predicted Values')
axs[1].set_title('Base model TEST: Actual vs. Predicted Values')

plt.tight_layout()
plt.show()

#Saving the trained model
import pickle
import joblib
joblib.dump(model_lr, 'model_lr.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

"""## Postproduction

*   Code for prediction
*   Test the prediction
*   Test different requirement methods

"""

#During prediction: preprocess the user input and predict
def predict_price(data):
    model = joblib.load('model_lr.pkl')  # Load the model
    preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor

    # Transform the data
    data = pd.DataFrame(data, index=[0])
    data = preprocessor.transform(data)

    # Predict
    prediction = model.predict(data)

    return prediction.tolist()

# loading the saved model
#loaded_model = pickle.load(open('get_around_model.sav', 'rb'))

# test the function
predict_price ({
  "car_model_name": "Citroën",
  "mileage": 13929,
  "engine_power": 150,
  "fuel": "petrol",
  "car_type": "convertible",
  "private_parking_available": 1,
  "has_gps": 1,
  "has_air_conditioning": 0,
  "automatic_car": 1,
  "has_getaround_connect": 0,
  "has_speed_regulator": 1,
  "winter_tires": 1
})

predict_price ({
  "car_model_name": "Citroën",
  "mileage": 128035,
  "engine_power": 135,
  "fuel": "diesel",
  "car_type": "convertible",
  "private_parking_available": True,
  "has_gps": True,
  "has_air_conditioning": False,
  "automatic_car": False,
  "has_getaround_connect": True,
  "has_speed_regulator": True,
  "winter_tires": True
})

# The curl command that you put in terminal
curl -i -H "Content-Type: application/json" -X POST -d '{
  "car_model_name": "Citroën",
  "mileage": 1035,
  "engine_power": 135,
  "fuel": "diesel",
  "car_type": "convertible",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": true,
  "winter_tires": true
}' https://mysterious-bayou-53960-eb9532454419.herokuapp.com/predict

curl -i -H "Content-Type: application/json" -X POST -d '{
  "car_model_name": "Citroën",
  "mileage": 10135,
  "engine_power": 105,
  "fuel": "diesel",
  "car_type": "convertible",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": true,
  "winter_tires": true
}' https://mysterious-bayou-53960-eb9532454419.herokuapp.com/predict

curl -i -H "Content-Type: application/json" -X POST -d '{
"car_model_name": "Citroën",
  "mileage": 1399,
  "engine_power": 150,
  "fuel": "petrol",
  "car_type": "convertible",
  "private_parking_available": 1,
  "has_gps": 1,
  "has_air_conditioning": 1,
  "automatic_car": 1,
  "has_getaround_connect": 0,
  "has_speed_regulator": 1,
  "winter_tires": 1
}' https://mysterious-bayou-53960-eb9532454419.herokuapp.com/predict

#get prediction using requests
import requests

response = requests.post("https://pricestimatorapp-07a16ed5554d.herokuapp.com/predict", json={
    "input": [[ "Toyota",1399,150,"petrol","convertible",1, 1, 1,1,0,1,1]]
})
print(response.json())