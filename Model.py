# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset (assuming you have a dataset with NPK values, Crop Name, and corresponding fertilizer recommendations)
# Replace "your_dataset.csv" with the actual file name
df = pd.read_csv("Fertilizer Prediction.csv")

# Print column names
print(df.columns)

# Display the first few rows of the dataset
df.head()

# Feature columns (NPK values)
X = df[['Nitrogen', 'Potassium', 'Phosphorous', 'Crop_Name']]

# Target column (Fertilizer recommendation)
y = df['Fertilizer Name']

# Initialize and train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X.drop('Crop_Name', axis=1), y)

# Save the trained model using pickle
with open('fertilizer_recommendation_model.pkl', 'wb') as pickle_out:
    pickle.dump(classifier, pickle_out)

# Example of how to use the model for prediction with user input NPK values and Crop Name
user_nitrogen = float(input("Enter Nitrogen value: "))
user_potassium = float(input("Enter Potassium value: "))
user_phosphorous = float(input("Enter Phosphorous value: "))
user_crop_name = input("Enter Crop Name: ")

user_input_values = np.array([[user_nitrogen, user_potassium, user_phosphorous, user_crop_name]])

if 'Crop_Name' in X.columns:
    user_recommendation = classifier.predict(user_input_values[:, :-1])  # Exclude Crop Name for prediction
    print(f"Recommended Fertilizer is: {user_recommendation[0]}")
else:
    user_recommendation = classifier.predict(user_input_values)
    print("Recommended Fertilizer is:", user_recommendation[0])
