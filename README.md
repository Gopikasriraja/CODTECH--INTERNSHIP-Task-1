
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'data.csv' with your actual file path)
data = pd.read_csv('data.csv')

# Check for missing values and handle them by replacing with column means
data.fillna(data.mean(numeric_only=True), inplace=True)

# Normalize/Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

print("Data processing completed.")
