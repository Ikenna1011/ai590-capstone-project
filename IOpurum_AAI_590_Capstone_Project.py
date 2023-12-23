# %%
# Improt libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# %%
# Import the dataset
df = pd.read_csv('SMS SITE 002, RIG-2_1min.csv')
df.head()

# %%
df.shape

# %%
# Descriptive statistics of the dataset
df.describe()

# %%
# Get the data types
df.info()

# %% [markdown]
# ## Data Cleaning

# %%
# Convert columns to numeric and removing the '%' sign
df['Sep1 BS&W %'] = pd.to_numeric(df['Sep1 BS&W %'].str.replace('%', ''))
df['BS&W at choke %'] = df['BS&W at choke %'].astype(str)
df['BS&W at choke %'] = pd.to_numeric(df['BS&W at choke %'].str.replace('%', ''))
df['Sep1 Gas CO2 %'] = pd.to_numeric(df['Sep1 Gas CO2 %'].str.replace('%', ''))

# Sort the data
df = df.sort_values('Time & Date')

# Converting to datetime format
df['Time & Date'] = pd.to_datetime(df['Time & Date'], format="%d/%m/%Y %H:%M", errors='coerce')

# Creating new columns
df['Year'] = pd.to_numeric(df['Time & Date'].dt.year, errors='coerce').astype('Int64')
df['Month'] = pd.to_numeric(df['Time & Date'].dt.month, errors='coerce').astype('Int64')
df['Day'] = pd.to_numeric(df['Time & Date'].dt.day, errors='coerce').astype('Int64')
df['Hour'] = pd.to_numeric(df['Time & Date'].dt.hour, errors='coerce').astype('Int64')
df['Minute'] = pd.to_numeric(df['Time & Date'].dt.minute, errors='coerce').astype('Int64')

# Fixing the NaN values
df['Year'] = df['Year'].fillna(method='ffill')
df['Month'] = df['Month'].fillna(method='ffill')
df['Day'] = df['Day'].fillna(method='ffill')
df['Hour'] = df['Hour'].fillna(method='ffill')
df['Minute'] = df['Minute'].fillna(method='ffill')

# Drop the 'Time & Date' column
df.drop('Time & Date', axis=1, inplace=True)

# Rearranging the columns in the dataset
df = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Primary Choke Size 64ths', 'WHP SMS PSIG', 
         'WHT  SMS/ECS (F)', 'DSDP/SFP PSIG', 'DSFP/UCP PSIG', 'UCT (F)', 'DCP PSIG', 'DCT/ECSDT (F)', 
         'Gas Static Pressure PSIG', 'Gas Temp (F)', 'Gas Diff Press Inches H20', 'Sep1 BS&W %',
         'Sep1 Gas Orifice Size Inches', 'Oil Temperature (F)', 'BS&W at choke %', 'Sep1 Gas CO2 %', 'Sep1 Gas H2S ppm',
       'Water Ph <none>', 'Chlorides PPM ppm', 'Sep1 Gas SG <none>', 'Sep1 Corr Oil SG API', 'Gas Flow MMSCF/Day', 
       'Oil Flow Bbls/Day', 'Water Flow Bbls/Day', 'TCA PSIG', 'CCA 9*13 PSIG', 'CCA 13*18 PSIG', 'Unnamed: 28', 
       'Unnamed: 29']]

# Checking the first 5 rows of the dataset
df.head()

# %%
# Displaying the total number of missing values
missing_values = df.isnull().sum()
percentage_missing = (missing_values / len(df)) * 100
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage Missing': percentage_missing
})
print(missing_data_summary)


# %%
# Visualize columns with missing values
plt.figure(figsize=(12, 6))
plt.title('Columns with Missing Values')
plt.xlabel('Columns')
plt.ylabel('Percentage Missing')
plt.xticks(rotation=90)
sns.barplot(x=missing_data_summary.index, y='Percentage Missing', data=missing_data_summary)
plt.show()


# %%
# Droped columns
df.drop('Unnamed: 28', axis=1, inplace=True)
df.drop('Unnamed: 29', axis=1, inplace=True)

# %%
# Removing special characters, lower case
df.columns = (
    df.columns
    .str.lower()
    .str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    .str.replace(r'_{2,}', '_', regex=True)
)
df.columns = [col.strip('_') for col in df.columns]
print(df)

# %%
# Data distribution plot
def  plot_dist(df, col):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.distplot(df[col], ax=ax[0])
    sns.boxplot(df[col], ax=ax[1])
    plt.show()
plt.show()
plot_dist(df, 'gas_flow_mmscf_day')

# %%
plot_dist(df, 'oil_flow_bbls_day')

# %%
plot_dist(df, 'water_flow_bbls_day')

# %%
# Dropping columns with zero variance is shown in the heatmap
df.drop('water_ph_none', axis=1, inplace=True)
df.drop('year', axis=1, inplace=True)

# %%
# Rmoving outliers
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df
df = remove_outliers(df, 'gas_flow_mmscf_day')
df = remove_outliers(df, 'oil_flow_bbls_day')
df = remove_outliers(df, 'water_flow_bbls_day')
df.shape

# %%
# Distribution plot after removing outliers
plot_dist(df, 'gas_flow_mmscf_day')
plot_dist(df, 'oil_flow_bbls_day')
plot_dist(df, 'water_flow_bbls_day')

# Create a heatmap
corr_matrix = df.corr()
plt.figure(figsize=(25, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.show()

# %% [markdown]
# ## PCA 

# %%
df.drop('month', axis=1, inplace=True)
df.drop('day', axis=1, inplace=True)
df.drop('hour', axis=1, inplace=True)
df.drop('minute', axis=1, inplace=True)

# Normalize data
df_norm = (df - df.mean()) / df.std()

# Create a PCA object
n_components = 5
pca = PCA(n_components=n_components)
pca.fit(df_norm)

# Transform the dataset
df_pca = pca.transform(df_norm)
print(df_pca.shape)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
df_pca.head()

# %% [markdown]
# ## Model Design, Building, Optimization and Analysis

# %%
X = df_pca
y = df[['gas_flow_mmscf_day', 'oil_flow_bbls_day', 'water_flow_bbls_day']]
y.head()

# Split the data
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=0)

# Check the shape of the training, validation and test sets
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# Train and evaluate the models
def train_and_evaluate(model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mae = np.mean(abs(predictions - y_val))
    
    # Performance metrics
    print('Performance Metrics for {}'.format(model.__class__.__name__))
    print('Mean Absolute Error: {:.4}'.format(mae))
    print('Root Mean Squared Error: {:.4}'.format(np.sqrt(np.mean((predictions - y_val) ** 2))))
    print('\n')
    
    # Return the performance metric
    return mae

# Train MLP model and evaluate its performance
mlp = MLPRegressor(random_state=0)
mlp_mae = train_and_evaluate(mlp)


# %%
# Making predictions on the test set
predictions = mlp.predict(X_test)
mae = np.mean(abs(predictions - y_test))
print('Performance Metrics for {}'.format(mlp.__class__.__name__))
print('Mean Absolute Error: {:.4}'.format(mae))
print('Root Mean Squared Error: {:.4}'.format(np.sqrt(np.mean((predictions - y_test) ** 2))))
print('\n')

# Plotting the actual values against the predicted values
plt.figure(figsize=(10, 5))
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.scatter(y_test['gas_flow_mmscf_day'], predictions[:, 0])
plt.scatter(y_test['oil_flow_bbls_day'], predictions[:, 1])
plt.scatter(y_test['water_flow_bbls_day'], predictions[:, 2])
plt.legend(['Gas Flow', 'Oil Flow', 'Water Flow'])
plt.show()

# %%
# Training and evaluating the RF model
rf = RandomForestRegressor(random_state=0)
rf_mae = train_and_evaluate(rf)

# %%
# Make predictions on the test set
predictions = rf.predict(X_test)
# Mean absolute error (MAE)
mae = np.mean(abs(predictions - y_test))
# Display the performance metrics
print('Performance Metrics for {}'.format(rf.__class__.__name__))
print('Mean Absolute Error: {:.4}'.format(mae))
print('Root Mean Squared Error: {:.4}'.format(np.sqrt(np.mean((predictions - y_test) ** 2))))
print('\n')

# %%
# Neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3)
])

learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mae',
    metrics=['mae'])
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True)

# Fit the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping])

# %%
# Model Evaluation
# Test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test MAE: {}'.format(test_mae))
# Validation set
val_loss, val_mae = model.evaluate(X_val, y_val)
print('Validation Loss: {}'.format(val_loss))
print('Validation MAE: {}'.format(val_mae))

# %%



