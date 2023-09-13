import streamlit as st
from amazon_bot import chat_with_bot
import joblib
ml_model=joblib.load("LinearSVC.pkl")
st.title('Amazon Chatbot')

user_input=st.sidebar.text_input("Enter your inquiry:")

if st.sidebar.button("Submit"):
    if user_input.lower() == 'quit':
        st.sidebar.text("Chatbot: Goodbye!")
    else:
        bot_response=chat_with_bot(user_input)
        ml_prediction=ml_model.predict([user_input])
        st.write("Amazon bot response:")
        st.write(bot_response)
        st.write("ML Model Prediction:")
        st.write(ml_prediction)
