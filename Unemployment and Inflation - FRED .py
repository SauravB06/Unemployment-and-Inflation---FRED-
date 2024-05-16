#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install fredapi')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.metrics import r2_score

plt.style.use('fivethirtyeight')
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

from fredapi import Fred

fred_key = '18f3cc1280bbcf08e7b100684f5e0eb3'
fred = Fred(fred_key)


# In[11]:


fred.search('unemployment rate')


# In[38]:


unrate = fred.get_series('UNRATE')


# In[40]:


unrate.plot()


# In[12]:


unemp_df = fred.search('unemployment rate state', filter=('frequency','Monthly'))
#cleaning data, to only use seasonally adjusted, monthly, and unit percent unemployment data
unemp_df = unemp_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')
#only use data that contains unemployment rate in the title
unemp_df = unemp_df.loc[unemp_df['title'].str.contains('Unemployment Rate')]


# In[13]:


unemp_df


# In[55]:


#grab the dataframe data from each dataset. Loop
len(unemp_df.index)


# In[14]:


all_results = []

for myid in unemp_df.index:
    results = fred.get_series(myid)
    results = results.to_frame(name=myid)
    all_results.append(results)
    time.sleep(0.1)


# In[76]:


unemp_results = pd.concat(all_results, axis=1).drop(['M08311USM156SNBR'], axis = 1)


# In[125]:


unemp_states = unemp_results.drop('UNRATE', axis = 1)


# In[100]:


unemp_states


# In[8]:


#plot states unemployment rate
px.line(unemp_states)


# In[387]:


#dropping these columns, because the columns with letters such as 'NVUR' stands for state initals + unemployment rate.
unemp_states_clean = unemp_states.copy()

# Iterate over column names and drop columns with names longer than 4 characters
for column in unemp_states.columns:
    if len(column) > 4:
        unemp_states_clean.drop(column, axis=1, inplace=True)

print(unemp_states_clean)


# In[388]:


px.line(unemp_states_clean)


# In[389]:


#making the title the state name
id_to_state = unemp_df['title'].str.replace('Unemployment Rate in','').to_dict()


# In[390]:


unemp_states_clean.columns = [id_to_state[c] for c in unemp_states_clean.columns]


# In[391]:


px.line(unemp_states_clean)


# In[392]:


#pulling the data from april 2020
ax = unemp_states_clean.loc[unemp_states.index == '2020-04-01'].T.sort_values('2020-04-01').plot(kind = 'barh', figsize = (20,20), title = "Unemployment Rate by State and U.S Territory - April of 2020", width = 0.7, edgecolor = 'black')
ax.legend().remove()
plt.show()


# In[325]:


#pull part. rate
part_df = fred.search('participation rate states', filter=('frequency','Monthly'))
part_df = part_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')


# In[296]:


type(part_df)


# In[345]:


#removing all of the data that isnt state participation rate 
sense = part_df['title'].str.contains('Labor Force Participation Rate for')
part_df_cleaned = part_df[sense]


# In[346]:


type(part_df_cleaned)


# In[349]:


part_df_result = []

for myid in part_df_cleaned.index:
    results = fred.get_series(myid)
    results = results.to_frame(name=myid)
    part_df_result.append(results)
    time.sleep(0.1)


# In[350]:


type(part_df_result)


# In[351]:


part_df_state = pd.concat(part_df_result, axis=1)


# In[352]:


px.line(part_df_state)


# In[335]:


part_df_cleaned


# In[353]:


part_df_state


# In[354]:


id = part_df_cleaned['title'].str.replace('Labor Force Participation Rate for','').to_dict()


# In[355]:


part_df_state.columns


# In[356]:


part_df_state.columns = [id[c] for c in part_df_state.columns]


# In[357]:


px.line(part_df_state)


# In[393]:


unemp_states_clean.columns


# In[394]:


unemp_states_clean = unemp_states_clean.drop(columns=[' Puerto Rico'])


# In[395]:


# Fix DC, overwrite
unemp_states_clean = unemp_states_clean.rename(columns = {' the District of Columbia' : ' District of Columbia'})


# In[462]:


#unemployment vs participation rate plot
fig, axs = plt.subplots(10, 5, figsize=(30, 30), sharex=True)
axs = axs.flatten()

i = 0
for state in unemp_states_clean.columns:
    if state == ' District of Columbia':
        continue
    ax2 = axs[i].twinx()
    unemp_states_clean.query('index >= 2020 and index < 2022')['' + state].plot(ax = axs[i], label = 'Unemployment')
    part_df_state.query('index >= 2020 and index < 2022')[''+ state].plot(ax = ax2, label = 'Participation', color = color_pal[1])
    axs[i].set_title(state)
    i += 1
plt.tight_layout()
plt.show()


# In[417]:


min_wage_df = fred.search('Sticky Price Consumer Price Index less Food')
#, filter=('frequency','Monthly'))
#part_df = part_df.query('seasonal_adjustment == "Seasonally Adjusted" and units == "Percent"')


# In[418]:


min_wage_df


# In[419]:


sticky_df = fred.get_series('CORESTICKM157SFRBATL')


# In[421]:


px.line(sticky_df)


# In[425]:


sticky_df.index = pd.to_datetime(sticky_df.index)

# Filter rows after 1976-01-01
sticky_df_filtered = sticky_df.loc['1976-01-01':]

print(sticky_df_filtered)


# In[427]:


px.line(sticky_df_filtered)


# In[449]:


correlation_results = {}

for column in unemp_states_clean.columns:
    correlation = sticky_df_filtered.corr(unemp_states_clean[column])
    correlation_results[column] = correlation

# Convert the results to a DataFrame for better visualization
correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index', columns=['Correlation'])
correlation_df.index.name = 'Series'

print(correlation_df)


# In[450]:


# Assuming correlation_df is your DataFrame containing correlation results

# Sort the DataFrame by correlation values
correlation_df_sorted = correlation_df.sort_values(by='Correlation', ascending=False)

# Plot the correlations
plt.figure(figsize=(12, 10))  # Increase figure size
bars = plt.barh(correlation_df_sorted.index, correlation_df_sorted['Correlation'], color='skyblue')

# Rotate and align the text labels
plt.yticks(rotation=0, ha='right', fontsize=15)  # Rotate labels to horizontal, align to the right

# Add values on the bars
for bar in bars:
    yval = bar.get_width()
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, round(yval, 2), va='center')

plt.xlabel('Correlation')
plt.title('Correlation between Sticky Price Consumer Price Index and Unemployment for each state')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[464]:


# Create an empty dictionary to store R^2 scores for each series
r_squared_scores = {}

# Iterate over each series in unemp_states_clean
for column in unemp_states_clean.columns:
    # Select the specific series
    specific_series = unemp_states_clean[column]
    
    # Drop NaN values from both sticky_df_filtered and the specific series
    combined_df = pd.concat([unemp_states_clean[column], sticky_df_filtered], axis=1).dropna()
    
    # Assign the columns appropriately for the regression
    unemployment_cleaned = combined_df.iloc[:, 0]
    inflation_cleaned = combined_df.iloc[:, 1]
    
    # Calculate R^2
    r_squared = r2_score(unemployment_cleaned, inflation_cleaned)
    
    # Store the R^2 score in the dictionary
    r_squared_scores[column] = r_squared

# Convert the dictionary to a DataFrame for better visualization
r_squared_df = pd.DataFrame.from_dict(r_squared_scores, orient='index', columns=['R^2 Score'])
r_squared_df.index.name = 'Series'

print(r_squared_df)

#these are not normal results. unemployment rate and inflation should have a strong coorelation


# In[463]:


# Align the indices of sticky_df_filtered and unemployment series
combined_df = pd.concat([sticky_df_filtered, unemp_states_clean], axis=1).dropna()

# Determine the number of rows and columns for the grid layout
num_series = len(unemp_states_clean.columns)
num_cols = 3  # Adjust as needed
num_rows = (num_series + num_cols - 1) // num_cols

# Create subplots with the specified grid layout
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Loop through each unemployment series
for i, column in enumerate(unemp_states_clean.columns):
    # Select the specific unemployment series from the aligned DataFrame
    unemployment_series = combined_df[column]
    
    # Create a scatter plot
    axs[i].scatter(combined_df.iloc[:, 0], unemployment_series, label='Data points')  # Use the first column of combined_df
    
    # Fit a linear regression line (inverse relationship)
    coefficients = np.polyfit(combined_df.iloc[:, 0], unemployment_series, 1)  # Use the first column of combined_df
    poly = np.poly1d(coefficients)
    axs[i].plot(combined_df.iloc[:, 0], poly(combined_df.iloc[:, 0]), color='red', label='Regression line')  # Use the first column of combined_df
    
    axs[i].set_xlabel('Inflation Rate')
    axs[i].set_ylabel('Unemployment Rate')
    axs[i].set_title(column)
    axs[i].legend(loc='upper right')

# Hide any empty subplots
for ax in axs[num_series:]:
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

