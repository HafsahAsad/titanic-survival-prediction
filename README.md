# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
data = pd.read_csv("titanic.csv")  # Load CSV file into a DataFrame

# Step 3: Drop columns that don't help much with prediction
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Step 4: Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Step 5: Encode categorical columns
label = LabelEncoder()
data["Sex"] = label.fit_transform(data["Sex"])           # male=1, female=0
data["Embarked"] = label.fit_transform(data["Embarked"]) # S=2, C=0, Q=1 (order may vary)

# Step 6: Define features and target variable
X = data.drop("Survived", axis=1)  # Features
y = data["Survived"]               # Target variable

# Step 7: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 10: Visualize feature importance
feature_importance = model.feature_importances_
plt.figure(figsize=(8,6))
sns.barplot(x=feature_importance, y=X.columns)
plt.title("Feature Importance")
plt.show()


