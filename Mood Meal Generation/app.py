import streamlit as st
import pandas as pd
import joblib

# Load model and data
model = joblib.load("mood_model.pkl")
recipes = pd.read_csv("mood_recipes.csv")

st.set_page_config(page_title="🍽️ Mood-Based Meal Generator", layout="centered")
st.title("🤖 Mood-Based Meal Generator")
st.markdown("Describe how you're feeling and I’ll suggest a meal!")

# Input mood text
user_mood = st.text_input("💬 How do you feel?", "I'm feeling lazy and want something easy")

if user_mood:
    prediction = model.predict([user_mood])[0]
    result = recipes[recipes["Recipe"] == prediction].iloc[0]

    st.subheader(f"🍲 Suggested Meal: {prediction}")
    st.caption(f"Ingredients: {result['Ingredients']}")
    st.success("Enjoy your personalized meal! 🧠")