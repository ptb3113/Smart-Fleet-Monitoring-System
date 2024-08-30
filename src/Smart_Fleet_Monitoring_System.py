#!/usr/bin/env python
# coding: utf-8

# # Smart Fleet Monitoring System
# 
# # By: Pratham Brahmbhatt
# 

# Project Name: Smart Fleet Monitoring System
# 
# Project Objective:
# The objective of the Smart Fleet Monitoring System is to develop a robust and scalable platform that utilizes real-time data processing, advanced machine learning models, and dynamic visualizations to monitor and optimize the health and performance of a vehicle fleet. The system aims to increase data processing efficiency, improve predictive maintenance accuracy, and enhance decision-making capabilities by leveraging tools such as Apache Spark, Airflow, SQL, and Tableau. This project also focuses on effective project management using JIRA, ensuring that all components are seamlessly integrated and aligned with industry best practices.
# 

# # 01. Data Processing and System Architecture

# Step 1: Set Up the Development Environment

# In[1]:


# Step 1.1: Install Required Libraries
get_ipython().system('pip install pandas dask sqlalchemy numpy')


# In[2]:


# Step 1.2: Import Necessary Libraries
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sqlalchemy import create_engine


# Step 2: Load and Process the Dataset

# In[3]:


# Step 2.1: Load Dataset into Pandas DataFrame
df = pd.read_csv('fleet_health_performance_dataset.csv')

# Display the first few rows of the dataset
df.head()


# 2.2 Basic Data Preprocessing with Pandas
# 
# 

# In[4]:


# Step 2.2.1: Handle Missing Values
df.fillna(method='ffill', inplace=True)


# In[5]:


# Step 2.2.2: Convert Timestamp to Datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[6]:


# Step 2.2.3: Convert Categorical Columns to Category Type
df['Vehicle ID'] = df['Vehicle ID'].astype('category')

# Display the DataFrame after preprocessing
df.head()


# 2.3 Upload Data to SQL Database

# In[7]:


# Step 2.3.1: Establish a Connection to the SQLite Database
import sqlite3

# Create a connection to the SQLite database
conn = sqlite3.connect('fleet_health_performance.db')


# In[8]:


# Step 2.3.2: Write DataFrame to SQLite Database
# Write the DataFrame to the SQLite database
df.to_sql('fleet_data', conn, if_exists='replace', index=False)


# In[9]:


# Step 2.3.3: Verify Data Load
# Query the database to verify the data load
result = pd.read_sql('SELECT * FROM fleet_data LIMIT 5;', conn)
print(result)


# Step 3: Data Processing with Dask and SQL

# In[10]:


# Step 3.1: Load Data from CSV into Dask DataFrame
import dask.dataframe as dd

# Load the data from the CSV file using Dask
dask_df = dd.read_csv('fleet_health_performance_dataset.csv')

# Show the first few rows to verify the load
dask_df.head()


# 3.2 Data Transformation with Dask

# In[11]:


# Step 3.2.1: Calculate Average Engine Temperature per Vehicle with Dask
avg_engine_temp = dask_df.groupby('Vehicle ID')['Engine Temp (°C)'].mean().compute()

# Display the result
print(avg_engine_temp)


# 3.3 Data Processing with SQL

# In[12]:


# Step 3.3.1: Query SQL Database for Average Engine Temperature
# Import SQLite3 for database operations
import sqlite3

# Establish a connection to the SQLite database
conn = sqlite3.connect('fleet_health_performance.db')

# Perform the query
query = """
SELECT "Vehicle ID", AVG("Engine Temp (°C)") as avg_engine_temp
FROM fleet_data
GROUP BY "Vehicle ID"
"""

# Execute the query and load the result into a Pandas DataFrame
avg_engine_temp_sql = pd.read_sql(query, con=conn)

# Display the result
print(avg_engine_temp_sql)

# Close the database connection
conn.close()


# Step 4: Automate Data Pipeline

# In[13]:


# Step 4.1: Import Required Libraries
import time
import sqlite3
import pandas as pd
import dask.dataframe as dd


# In[14]:


# Step 4.2: Define the Data Processing Function
def preprocess_and_query_data():
    print("Starting data preprocessing and SQL queries...")

    # Load the data into Dask (or Pandas for smaller datasets)
    dask_df = dd.read_csv('fleet_health_performance_dataset.csv')

    # Perform a transformation (e.g., calculate average engine temperature)
    avg_engine_temp = dask_df.groupby('Vehicle ID')['Engine Temp (°C)'].mean().compute()
    print("Average Engine Temperature Calculation Done.")

    # Example SQL operation
    conn = sqlite3.connect('fleet_health_performance.db')
    query = """
    SELECT "Vehicle ID", AVG("Engine Temp (°C)") as avg_engine_temp
    FROM fleet_data
    GROUP BY "Vehicle ID"
    """
    avg_engine_temp_sql = pd.read_sql(query, con=conn)
    print("SQL Query Done.")
    print(avg_engine_temp_sql)
    conn.close()

    print("Data processing and SQL queries completed.")


# In[ ]:


# Step 4.3: Run the Task in a Loop
while True:
    preprocess_and_query_data()
    print("Task completed. Waiting for the next run...")
    time.sleep(86400)  # Sleep for 24 hours (86400 seconds)


# In[ ]:





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





# # 03. Dashboard Design and Visualization

# Step 1: Data Preparation for Visualization

# In[ ]:


# Step 1.1: Import Required Libraries
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Step 1.2: Load Data from the SQLite Database
conn = sqlite3.connect('fleet_health_performance.db')

# Load the entire dataset or specific parts needed for visualization
df = pd.read_sql('SELECT * FROM fleet_data', conn)

# Close the database connection
conn.close()

# Display the first few rows of the DataFrame
df.head()


# In[ ]:


# Step 1.3: Prepare Data for Visualization
# Example: Convert timestamps to datetime if not already done
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Example: Create additional columns or aggregate data as needed
avg_temp_per_vehicle = df.groupby('Vehicle ID')['Engine Temp (°C)'].mean().reset_index()


# Step 2: Create Static Visualizations with Matplotlib and Seaborn

# In[ ]:


# Step 2.1: Plot Engine Temperature Over Time for a Specific Vehicle
plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['Engine Temp (°C)'])
plt.title('Engine Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Engine Temperature (°C)')
plt.show()


# In[ ]:


# Step 2.2: Plot Average Engine Temperature by Vehicle
plt.figure(figsize=(10, 6))
sns.barplot(x='Vehicle ID', y='Engine Temp (°C)', data=avg_temp_per_vehicle)
plt.title('Average Engine Temperature by Vehicle')
plt.xlabel('Vehicle ID')
plt.ylabel('Average Engine Temperature (°C)')
plt.show()


# In[ ]:


# Step 2.3: Create a Heatmap to Show Correlations Between Features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Step 3: Build Interactive Dashboards Using Plotly and Dash

# In[ ]:


# Step 3.1: Install Plotly and Dash
get_ipython().system('pip install plotly dash')


# In[ ]:


# Step 3.2: Set Up a Basic Dash App
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Fleet Health Monitoring Dashboard"),
    dcc.Dropdown(
        id='vehicle-dropdown',
        options=[{'label': vehicle, 'value': vehicle} for vehicle in df['Vehicle ID'].unique()],
        value=df['Vehicle ID'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='temp-graph')
])

# Define the callback to update the graph based on the selected vehicle
@app.callback(
    Output('temp-graph', 'figure'),
    [Input('vehicle-dropdown', 'value')]
)
def update_graph(selected_vehicle):
    filtered_df = df[df['Vehicle ID'] == selected_vehicle]
    fig = px.line(filtered_df, x='Timestamp', y='Engine Temp (°C)',
                  title=f'Engine Temperature Over Time for Vehicle {selected_vehicle}')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:


# Step 3.3: Add Additional Interactive Elements to the Dashboard
app.layout = html.Div([
    html.H1("Fleet Health Monitoring Dashboard"),
    dcc.Dropdown(
        id='vehicle-dropdown',
        options=[{'label': vehicle, 'value': vehicle} for vehicle in df['Vehicle ID'].unique()],
        value=df['Vehicle ID'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='temp-graph'),
    dcc.Graph(id='battery-graph')
])

# Update both graphs based on the selected vehicle
@app.callback(
    [Output('temp-graph', 'figure'),
     Output('battery-graph', 'figure')],
    [Input('vehicle-dropdown', 'value')]
)
def update_graphs(selected_vehicle):
    filtered_df = df[df['Vehicle ID'] == selected_vehicle]
    temp_fig = px.line(filtered_df, x='Timestamp', y='Engine Temp (°C)',
                       title=f'Engine Temperature Over Time for Vehicle {selected_vehicle}')
    battery_fig = px.line(filtered_df, x='Timestamp', y='Battery SoC (%)',
                          title=f'Battery State of Charge Over Time for Vehicle {selected_vehicle}')
    return temp_fig, battery_fig

if __name__ == '__main__':
    app.run_server(debug=True)


# Step 4: Deploying the Dashboard

# In[ ]:


# Step 4.1: Deploying on a Local Server
# This is done by running the script above and accessing the dashboard via a local browser (e.g., http://127.0.0.1:8050/)


# In[ ]:





# # 04. Post-Deployment Monitoring and Maintenance
# 

# Step 1: System Monitoring

# In[ ]:


# Step 1.1: Import Required Libraries
import time
import logging
from datetime import datetime
import psutil  # For system performance monitoring


# In[ ]:


# Step 1.2: Set Up Logging
logging.basicConfig(filename='system_monitor.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


# In[ ]:


# Step 1.3: Monitor System Performance
def monitor_system():
    while True:
        # Monitor CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        logging.info(f"CPU Usage: {cpu_usage}%")
        
        # Monitor Memory usage
        memory = psutil.virtual_memory()
        logging.info(f"Memory Usage: {memory.percent}%")
        
        # Monitor Disk usage
        disk_usage = psutil.disk_usage('/')
        logging.info(f"Disk Usage: {disk_usage.percent}%")

        # Monitor Network activity
        net = psutil.net_io_counters()
        logging.info(f"Bytes Sent: {net.bytes_sent}, Bytes Received: {net.bytes_recv}")

        # Sleep for a defined interval before checking again
        time.sleep(3600)  # Monitor every hour

# Start monitoring in the background
monitor_system()


# Step 2: Model Retraining and Updates

# In[ ]:


# Step 2.1: Import Required Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd


# In[ ]:


# Step 2.2: Load New Data and Prepare for Retraining
def load_and_prepare_data():
    # Load the new dataset
    df = pd.read_csv('new_fleet_health_performance_data.csv')
    
    # Feature engineering (same steps as Part 2)
    df['Vehicle Age (Days)'] = (pd.to_datetime('today') - pd.to_datetime(df['Timestamp'])).dt.days
    df['Engine Temp x Battery SoC'] = df['Engine Temp (°C)'] * df['Battery SoC (%)']
    
    # Select features and target
    features = df[['Engine Temp (°C)', 'Battery SoC (%)', 'Tire Pressure (psi)',
                   'Distance Traveled (km)', 'Vehicle Age (Days)', 'Engine Temp x Battery SoC']]
    target = df['Predicted Failure (Days)']
    
    return train_test_split(features, target, test_size=0.2, random_state=42)


# In[ ]:


# Step 2.3: Retrain the Model
def retrain_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Load the existing model
    model = joblib.load('best_fleet_model.pkl')
    
    # Retrain the model with new data
    model.fit(X_train, y_train)

    # Evaluate the retrained model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Retrained Model MSE: {mse}")

    # Save the retrained model
    joblib.dump(model, 'best_fleet_model.pkl')
    logging.info("Retrained model saved.")

# Schedule model retraining every month
while True:
    retrain_model()
    time.sleep(2592000)  # Retrain every 30 days (2592000 seconds)


# Step 3: Continuous Improvement

# In[ ]:


# Step 3.1: Incorporate User Feedback
def gather_user_feedback():
    # Simulate feedback collection (e.g., from a form or database)
    feedback = {
        'feature_requests': ['Add GPS tracking', 'Improve battery monitoring'],
        'issues_reported': ['Slow dashboard loading', 'Inaccurate temperature readings']
    }
    
    logging.info(f"User Feedback: {feedback}")
    return feedback


# In[ ]:


# Step 3.2: Optimize System Based on Feedback
def optimize_system():
    feedback = gather_user_feedback()
    
    # Example: Adjust system based on feedback
    if 'Slow dashboard loading' in feedback['issues_reported']:
        logging.info("Optimization: Streamlining dashboard queries for faster loading.")
        # Implement actual optimization here

    if 'Improve battery monitoring' in feedback['feature_requests']:
        logging.info("Optimization: Enhancing battery monitoring feature.")
        # Implement actual enhancement here

# Run optimization periodically
while True:
    optimize_system()
    time.sleep(604800)  # Optimize every week (604800 seconds)


# In[ ]:





# In[ ]:




