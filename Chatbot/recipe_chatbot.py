import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia
import numpy as np


nlp = spacy.load("en_core_web_sm")

# Load dataset
try:
    df = pd.read_csv(r"C:\Users\bharg\Ankith\Python\Chatbot\datasets\Recipes.csv")
except FileNotFoundError:
    st.error("recipes.csv not found! Please place it in the project directory.")
    st.stop()

def extract_ingredients(text):
    """Extract potential ingredients from user input using spaCy."""
    doc = nlp(text.lower())
    ingredients = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ingredients

def find_recipes(budget, user_ingredients, dietary):
    """Find recipes matching budget, ingredients, and dietary preferences."""
    if dietary:
        dietary = dietary.lower()
        filtered_df = df[df["dietary"].str.lower().str.contains(dietary, na=False)]
    else:
        filtered_df = df

    filtered_df = filtered_df[filtered_df["cost"] <= budget]

    if not user_ingredients:
        return filtered_df

    # Use TF-IDF to match ingredients
    vectorizer = TfidfVectorizer()
    all_ingredients = filtered_df["ingredients"].tolist() + [", ".join(user_ingredients)]
    tfidf_matrix = vectorizer.fit_transform(all_ingredients)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarities
    filtered_df = filtered_df.sort_values("similarity", ascending=False).head(3)
    return filtered_df

def get_wiki_info(dish):
    """Fetch brief Wikipedia info for an ingredient."""
    try:
        summary = wikipedia.summary(dish, sentences=1)
        return f"**{dish.capitalize()}**: {summary}"
    except:
        return f"**{dish.capitalize()}**: No Wikipedia info found."

def generate_tips():
    """Provide grocery-saving tips."""
    return [
        "Buy in bulk (e.g., rice, pasta) to save on unit costs.",
        "Shop at discount stores or farmers' markets for fresh produce.",
        "Use frozen vegetables for longer shelf life and lower cost.",
        "Plan meals weekly to avoid impulse buys."
    ]

def generate_shopping_list(recipe_ingredients, user_ingredients):
    """Generate a shopping list of missing ingredients."""
    recipe_ings = [ing.strip() for ing in recipe_ingredients.split(",")]
    missing = [ing for ing in recipe_ings if ing.lower() not in [u.lower() for u in user_ingredients]]
    return missing

# Streamlit UI
st.title("Student Recipe Chatbot")
st.markdown("Find quick, low-cost recipes based on your budget, ingredients, and dietary preferences!")

# User inputs
budget = st.number_input("Budget (Rs)", min_value=0.0, max_value=800.0, value=150.0, step=50.0)
ingredients_input = st.text_area("Available Ingredients (e.g., rice, eggs)", placeholder="Type ingredients or leave blank")
dietary = st.selectbox("Dietary Preference", ["None", "Vegetarian", "Vegan","Low-carb","High-protein","Non-vegetarian","Gluten-free","Pescatarian","Dairy-free"], index=0)

if st.button("Find Recipes"):
    if dietary == "None":
        dietary = ""
    user_ingredients = extract_ingredients(ingredients_input) if ingredients_input else []

    # Find matching recipes
    recipes = find_recipes(budget, user_ingredients, dietary)
    
    if recipes.empty:
        st.warning("No recipes found! Try increasing your budget or changing preferences.")
    else:
        st.subheader("Recommended Recipes")
        for _, row in recipes.iterrows():
            st.write(f"**{row['name']}** (Cost: Rs{row['cost']:.2f}, Time: {row['prep_time']} min)")
            st.write(f"**Ingredients**: {row['ingredients']}")
            st.write(f"**Steps**:")
            for step in row['steps'].split(". "):
                if step.strip():
                    st.write(f"- {step.strip()}.")
            
            # Wikipedia info for first dish
            first_ing = row['name'].split(",")[0].strip()
            st.write(get_wiki_info(first_ing))
            
            # Shopping list
            missing = generate_shopping_list(row['ingredients'], user_ingredients)
            if missing:
                st.write(f"**Shopping List**: {', '.join(missing)}")
            else:
                st.write("**Shopping List**: You have all ingredients!")
            st.markdown("---")

        # Grocery-saving tips
        st.subheader("Grocery-Saving Tips")
        for tip in generate_tips():
            st.write(f"- {tip}")

# Footer
st.markdown("---")
st.write("Built by Ankith Krishna Bhargav.")
