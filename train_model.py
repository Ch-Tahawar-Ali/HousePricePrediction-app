# %%
import pandas as pd
import numpy as np

# %%
import seaborn as sns
import matplotlib as plt

# %%
from joblib import dump

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.metrics import r2_score,recall_score,precision_score,mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# %%
data=pd.read_csv("housing.csv")
data.head()

# %%
x=data.drop(columns=['median_house_value'])
y=data['median_house_value']

# %%
catagorical_features=['ocean_proximity']
numerical_features=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']

# %%
def add_derived_features(data):
    data = data.copy()
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data

# %%
numerical_features_transformer = Pipeline(steps=[
    ('add_derived_features', FunctionTransformer(add_derived_features)),
    ('imputeMissing', SimpleImputer(strategy='median')),
    ('stranderedscaler', StandardScaler())
])

# %%
catagorical_features_transformer = Pipeline(steps=[
    ('imputeMissing', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

# %%
preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_features_transformer, numerical_features),
    ('catagorical', catagorical_features_transformer, catagorical_features)
])

# %%
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# %%
final_pipeline.fit(x_train, y_train)

# %%
y_pred=final_pipeline.predict(x_test)

# %%
print("R2 Score:",  r2_score(y_test, y_pred))
print("Root Mean Square Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))

# %%
new_data=pd.DataFrame({
    'longitude':[-122.23],
    'latitude':[37.88],
    'housing_median_age':[41],
    'total_rooms':[880],
    'total_bedrooms':[129],
    'population':[322],
    'households':[126],
    'median_income':[8.3252],
    'ocean_proximity':['NEAR BAY']
})

# %%
predicted_value=final_pipeline.predict(new_data)
print("Predicted value:", predicted_value)

# %%
print("predicted Median House Value:",predicted_value)

# %%
dump(final_pipeline, "model_pipeline.joblib")


