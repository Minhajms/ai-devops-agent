import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = pd.read_csv("data/pipeline_data.csv")

# Preprocess data
X = data.drop("failure", axis=1)  # Features
y = data["failure"]  # Target

# Train a model
logger.info("Training model...")
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "pipeline_model.pkl")
logger.info("Model trained and saved!")
