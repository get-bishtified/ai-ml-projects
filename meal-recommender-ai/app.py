import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load recipe data
@st.cache_data
def load_data():
    data = pd.read_csv("recipes.csv")
    return data

data = load_data()

# Prepare TF-IDF vector
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Ingredients'])

# UI
st.set_page_config(page_title="🍽️ What Should I Eat?", layout="centered")
st.title("🤖 What Should I Eat?")
st.markdown("Enter ingredients you have, and get instant AI meal ideas!")

user_input = st.text_input("🥕 Ingredients (comma-separated)", "tomato, onion, garlic")

if user_input:
    # Predict
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)

    top_indices = similarity[0].argsort()[-3:][::-1]
    st.subheader("🍲 Meal Suggestions:")
    for idx in top_indices:
        st.markdown(f"- **{data.iloc[idx]['Recipe']}**")
        st.caption(f"Ingredients: {data.iloc[idx]['Ingredients']}")

    st.success("Bon appétit! 🎉")