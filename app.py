import streamlit as st
from amazon_bot import chat_with_bot
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained ML model and TF-IDF vectorizer
ml_model = joblib.load("model.pkl")
tfidf_vectorizer = joblib.load("tfidf.pkl")  # Replace with the actual filename

st.title('Amazon Chatbot')

user_input = st.sidebar.text_input("Enter your inquiry:")

if st.sidebar.button("Submit"):
    if user_input.lower() == 'quit':
        st.sidebar.text("Chatbot: Goodbye!")
    else:
        bot_response = chat_with_bot(user_input)
        
        # Preprocess user input using the loaded TF-IDF vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Make predictions using the loaded ML model
        ml_prediction = ml_model.predict(user_input_tfidf)
        
        st.write("Amazon bot response:")
        st.write(bot_response)
        st.write("ML Model Prediction:")
        st.write(ml_prediction)
