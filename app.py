from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class Message(BaseModel):
    message: str
 
@app.post("/predict/spam")
def predict_spam(msg: Message):
    # Load the trained model and vectorizer
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Transform the input message using the vectorizer
    message_vector = vectorizer.transform([msg.message.lower()])
    
    # Make prediction
    prediction = model.predict(message_vector)
    
    # Map prediction to label
    label = 'spam' if prediction[0] == 1 else 'Not spam'
    
    return {"label": label}