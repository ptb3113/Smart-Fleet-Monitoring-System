#!/usr/bin/env python
# coding: utf-8

# # 01. Data Processing and System Architecture

# Step 1: Set Up the Development Environment

# In[2]:


# Step 1.1: Install Required Libraries
get_ipython().system('pip install pandas dask sqlalchemy numpy')


# In[3]:


# Step 1.2: Import Necessary Libraries
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sqlalchemy import create_engine


# Step 2: Load and Process the Dataset

# In[ ]:


# Step 2.1: Load Dataset into Pandas DataFrame
df = pd.read_csv('fleet_health_performance_dataset.csv')

# Display the first few rows of the dataset
df.head()


# 2.2 Basic Data Preprocessing with Pandas
# 
# 

# In[ ]:


# Step 2.2.1: Handle Missing Values
df.fillna(method='ffill', inplace=True)


# In[ ]:


# Step 2.2.2: Convert Timestamp to Datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[ ]:


# Step 2.2.3: Convert Categorical Columns to Category Type
df['Vehicle ID'] = df['Vehicle ID'].astype('category')

# Display the DataFrame after preprocessing
df.head()


# 2.3 Upload Data to SQL Database

# In[ ]:


# Step 2.3.1: Establish a Connection to the SQLite Database
import sqlite3

# Create a connection to the SQLite database
conn = sqlite3.connect('fleet_health_performance.db')


# In[ ]:


# Step 2.3.2: Write DataFrame to SQLite Database
# Write the DataFrame to the SQLite database
df.to_sql('fleet_data', conn, if_exists='replace', index=False)


# In[ ]:


# Step 2.3.3: Verify Data Load
# Query the database to verify the data load
result = pd.read_sql('SELECT * FROM fleet_data LIMIT 5;', conn)
print(result)


# Step 3: Data Processing with Dask and SQL

# In[ ]:


# Step 3.1: Load Data from CSV into Dask DataFrame
import dask.dataframe as dd

# Load the data from the CSV file using Dask
dask_df = dd.read_csv('fleet_health_performance_dataset.csv')

# Show the first few rows to verify the load
dask_df.head()


# 3.2 Data Transformation with Dask

# In[ ]:


# Step 3.2.1: Calculate Average Engine Temperature per Vehicle with Dask
avg_engine_temp = dask_df.groupby('Vehicle ID')['Engine Temp (°C)'].mean().compute()

# Display the result
print(avg_engine_temp)


# 3.3 Data Processing with SQL

# In[ ]:


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

# In[ ]:


# Step 4.1: Import Required Libraries
import time
import sqlite3
import pandas as pd
import dask.dataframe as dd


# In[ ]:


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





# In[ ]:




