import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Gmail Spam Detection App")
st.write("Enter a message below to check if it's **Spam** or **Ham**:")

# Text input box
user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and predict
        message_vector = vectorizer.transform([user_input.lower()])
        prediction = model.predict(message_vector)[0]
        prediction_label = "Spam" if prediction == 1 else "Ham"
        probability = model.predict_proba(message_vector)[0]
        confidence = max(probability) * 100

        # Output
        st.success(f"Prediction: {prediction_label}")
        st.markdown(f"**Confidence : {confidence}**")
