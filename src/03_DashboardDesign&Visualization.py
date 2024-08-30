#!/usr/bin/env python
# coding: utf-8

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




