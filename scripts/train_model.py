import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("data/pipeline_data.csv")

# Preprocess data
X = data.drop("failure", axis=1)  # Features
y = data["failure"]  # Target

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "pipeline_model.pkl")
print("Model trained and saved!")
