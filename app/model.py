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
#from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import joblib
import pickle

df_price = pd.read_csv('/Users/karinapopova/Desktop/JEDHA/M11_CERTIF/C_depl/get_around_pricing_project.csv')
df_price.head(5)

#preprocessing
#feature engeneering
#rename a column 
df_price['car_model_name']= df_price['model_key']
df_price.drop(columns = ['model_key'], inplace = True )


#drop useless columns
df_price.drop(columns=['Unnamed: 0', 'paint_color'], inplace=True)
# convert booleans into integers
df_price.replace({False: 0, True: 1}, inplace=True)

#DROP OUTLIERS
columns_to_check = ['mileage', 'engine_power']
# Calculate the lower and upper bounds for outliers
lower_val = df_price[columns_to_check].mean() - 3 * df_price[columns_to_check].std()
upper_val = df_price[columns_to_check].mean() + 3 * df_price[columns_to_check].std()
# Create a boolean mask for outlier rows
outlier_mask = ~((df_price[columns_to_check] < lower_val) | (df_price[columns_to_check] > upper_val)).any(axis=1)
# Filter the df to include rows without outliers
df_filtered = df_price[outlier_mask]
#print(f'There are {len(df_filtered)} rows that do not have outliers.')
#print(f'There are {len(df_price)- len(df_filtered)} rows to drop, which represents {round(100-((len(df_filtered)*100 )/len(df_price)), 2)} % of the original dataset. ')
print (df_filtered.head(5))

# Separate the target value "rental_price_per_day" from other features
target_name = 'rental_price_per_day'
Y = df_filtered.loc[:,target_name]
X = df_filtered.drop(target_name, axis = 1)

# Perform a train / test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

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

# Preprocessings on train set and on test set
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train model
model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
# Predictions on training set and on test set
Y_train_pred = model_lr.predict(X_train)
Y_test_pred = model_lr.predict(X_test)

# Explore metrics
print("MSE score on training set : ", mean_squared_error(Y_train, Y_train_pred))
print("MSE score on test set : ", mean_squared_error(Y_test, Y_test_pred))
print("R2 score on training set : ", r2_score(Y_train, Y_train_pred))
print("R2 score on test set : ", r2_score(Y_test, Y_test_pred))

# explore feature importance
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


# save model as pickle file
joblib.dump(model_lr, 'model_lr.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
