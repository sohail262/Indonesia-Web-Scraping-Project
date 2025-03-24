import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load the saved TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# ✅ Load the trained sentiment model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a sentence below to predict its sentiment.")

# User input box
user_input = st.text_area("Enter your sentence here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        #  Transform user input using the saved TF-IDF vectorizer
        X_input = tfidf.transform([user_input])

        #  Convert to DataFrame with matching feature names
        X_input_df = pd.DataFrame(X_input.toarray(), columns=tfidf.get_feature_names_out())

        #  Ensure columns are in the correct order
        X_input_df = X_input_df.reindex(columns=tfidf.get_feature_names_out(), fill_value=0)

        #  Convert DataFrame to NumPy array (Fix for RandomForestClassifier)
        X_input_np = X_input_df.to_numpy()

        #  Predict sentiment using trained model
        prediction = model.predict(X_input_np)[0]
        probabilities = model.predict_proba(X_input_np)

        #  Display the result
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        st.success(f"Predicted Prob: {probabilities}")
        st.success(f"Predicted Sentiment: {sentiment_label}")
