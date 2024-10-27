import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import ppscore as pps

# 1. Load the dataset
data = pd.read_csv('adult_with_headers.csv')

# 2. Basic exploration
print(data.info())        # Data types, missing values, etc.
print(data.describe())    # Summary statistics

# 3. Handle Missing Values (assuming none in your dataset as per the example, but here's a sample approach)
# Example: Fill missing values with the mode for categorical data
# (use data.isnull().sum() to check missing values and handle appropriately if present)
data.fillna(data.mode().iloc[0], inplace=True)

# 4. Apply Scaling
# Apply Standard Scaling to numerical columns
scaler = StandardScaler()
numerical_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Apply Min-Max Scaling to the same numerical columns for comparison
min_max_scaler = MinMaxScaler()
data_minmax = data.copy()
data_minmax[numerical_columns] = min_max_scaler.fit_transform(data_minmax[numerical_columns])

# 5. Encoding Categorical Variables
# Apply Label Encoding for categorical features with many categories
label_encoder = LabelEncoder()
for col in ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']:
    data[col] = label_encoder.fit_transform(data[col])

# 6. Feature Engineering
# Create a new feature for 'Work hours per year'
data['Work_hours_per_year'] = data['hours_per_week'] * 52

# Apply log transformation to skewed features (e.g., capital_gain)
data['log_capital_gain'] = np.log1p(data['capital_gain'])  # Log transform to handle skewness

# 7. Isolation Forest for Outlier Detection
iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination according to your data
outliers = iso_forest.fit_predict(data[numerical_columns])
data_cleaned = data[outliers == 1]  # Filter out outliers

# 8. PPS Analysis
pps_matrix = pps.matrix(data_cleaned)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
print(pps_matrix)

# Compare with the correlation matrix
corr_matrix = data_cleaned.corr()
print(corr_matrix)
