import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
recipes = pd.read_csv("mood_recipes.csv")

# Create mood classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=500))
])

# Train model on mood labels
model.fit(recipes["Mood"], recipes["Recipe"])

# Save the model
joblib.dump(model, "mood_model.pkl")
print("✅ Model trained and saved as mood_model.pkl")