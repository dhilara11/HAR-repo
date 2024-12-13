import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
file_path = "C:/intelligence/Human_Activity_Recognition_Using_Smartphones_Data.xlsx"
data = pd.read_excel(file_path)

# Display dataset information
print("Dataset Overview:")
print(data.info())  # Shows the structure and column details
print("\nFirst 5 Rows of the Dataset:")
print(data.head())  # Displays the first few rows of the dataset

# Remove duplicates if any
initial_rows = data.shape[0]
data = data.drop_duplicates()
print(f"\nRemoved {initial_rows - data.shape[0]} duplicate rows.")

# Handle missing data
print("\nChecking for Missing Values Per Column:")
print(data.isnull().sum())  # Check for missing values

# Fill missing values with column mean
data = data.fillna(data.mean())
print("Missing values have been handled.\n")

# Plot the distribution of activities
plt.figure(figsize=(10, 6))
sns.countplot(x='Activity', data=data, palette='viridis')
plt.title("Distribution of Activities")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Encode the categorical 'Activity' column
label_encoder = LabelEncoder()
data['Activity'] = label_encoder.fit_transform(data['Activity'])
print("Encoded Activity Labels:")
print(data['Activity'].unique())  # Display unique encoded labels


# Split features (X) and target labels (y)
X = data.drop('Activity', axis=1)  # Drop 'Activity' column as features
y = data['Activity']  # 'Activity' column is the target

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nData scaling completed.")

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

import joblib
# Save the trained model
joblib.dump(model, "trained_model.pkl")

# Later, load the model for reuse
loaded_model = joblib.load("trained_model.pkl")