#!/usr/bin/env python
# coding: utf-8

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




