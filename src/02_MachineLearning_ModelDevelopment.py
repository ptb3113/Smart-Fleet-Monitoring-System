#!/usr/bin/env python
# coding: utf-8

# # 02. Machine Learning Model Development

# Step 1: Feature Engineering

# In[ ]:


# Step 1.1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Step 1.2: Load the Dataset
df = pd.read_csv('fleet_health_performance_dataset.csv')

# Display the first few rows of the dataset
df.head()


# In[ ]:


# Step 1.3: Feature Engineering - Create New Features
# Example: Creating a feature for the vehicle's age in days
df['Vehicle Age (Days)'] = (pd.to_datetime('today') - pd.to_datetime(df['Timestamp'])).dt.days

# Example: Interaction terms or polynomial features
df['Engine Temp x Battery SoC'] = df['Engine Temp (°C)'] * df['Battery SoC (%)']

# Display the updated DataFrame with new features
df.head()


# In[ ]:


# Step 1.4: Select Features and Target Variable
features = df[['Engine Temp (°C)', 'Battery SoC (%)', 'Tire Pressure (psi)',
              'Distance Traveled (km)', 'Vehicle Age (Days)', 'Engine Temp x Battery SoC']]
target = df['Predicted Failure (Days)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features (important for certain models like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 2: Model Selection and Training

# In[ ]:


# Step 2.1: Import Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


# Step 2.2: Train a Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate the model
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

# Display results
print(f"Linear Regression MAE: {lr_mae}")
print(f"Linear Regression MSE: {lr_mse}")
print(f"Linear Regression R2: {lr_r2}")


# In[ ]:


# Step 2.3: Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

# Display results
print(f"Random Forest MAE: {rf_mae}")
print(f"Random Forest MSE: {rf_mse}")
print(f"Random Forest R2: {rf_r2}")


# In[ ]:


# Step 2.4: Train a Support Vector Regressor (SVR)
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svr = svr_model.predict(X_test_scaled)

# Evaluate the model
svr_mae = mean_absolute_error(y_test, y_pred_svr)
svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_r2 = r2_score(y_test, y_pred_svr)

# Display results
print(f"SVR MAE: {svr_mae}")
print(f"SVR MSE: {svr_mse}")
print(f"SVR R2: {svr_r2}")


# Step 3: Model Evaluation and Optimization

# In[ ]:


# Step 3.1: Compare Model Performance
models = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'SVR'],
    'MAE': [lr_mae, rf_mae, svr_mae],
    'MSE': [lr_mse, rf_mse, svr_mse],
    'R2 Score': [lr_r2, rf_r2, svr_r2]
})

# Display the comparison
print(models.sort_values(by='R2 Score', ascending=False))


# In[ ]:


# Step 3.2: Select the Best Model for Deployment
# Assuming Random Forest is the best model based on R2 Score
best_model = rf_model

# Save the model for deployment (if needed)
import joblib
joblib.dump(best_model, 'best_fleet_model.pkl')

print("Best model saved as 'best_fleet_model.pkl'")


# In[ ]:


# Step 3.3: Hyperparameter Tuning with GridSearchCV (Optional)
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from GridSearch
best_grid_model = grid_search.best_estimator_

# Evaluate the best grid model
y_pred_grid = best_grid_model.predict(X_test_scaled)
grid_mae = mean_absolute_error(y_test, y_pred_grid)
grid_mse = mean_squared_error(y_test, y_pred_grid)
grid_r2 = r2_score(y_test, y_pred_grid)

# Display the tuned model's performance
print(f"Tuned Random Forest MAE: {grid_mae}")
print(f"Tuned Random Forest MSE: {grid_mse}")
print(f"Tuned Random Forest R2: {grid_r2}")


# In[ ]:




