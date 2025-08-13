# Step 1: Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib  # To save the model

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Step 3: Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose and Train the Model
# You can choose either LogisticRegression or RandomForestClassifier

# Logistic Regression
# model = LogisticRegression(max_iter=200)

# OR use Random Forest (better for beginners)
model = RandomForestClassifier()

model.fit(X_train, y_train)  # Train the model

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Check Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Step 7: Save the model to a file using joblib
joblib.dump(model, 'iris_model.pkl')

print("Model saved as iris_model.pkl")
