# Step 1: Import Libraries and Load the Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
# Restrict word index to the top 10000 words
vocab_size = 10000
word_index = {word: idx for word, idx in word_index.items() if idx < vocab_size}
reverse_word_index = {idx: word for word, idx in word_index.items()}


reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
# Ensure the path is correct
model_path = r'Basic_Rnn_project/simple_rnn_model.keras'
model = load_model(model_path)

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    vocab_size = 10000 
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Default to index 2 for unknown words
    encoded_review = [idx if idx < vocab_size else 2 for idx in encoded_review]  # Clip to valid range
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Match model input shape
    return padded_review


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User Input with increased text area size
user_input = st.text_area('Movie Review', height=200)

if st.button('Classify'):
    if not user_input.strip():
        st.write('Please enter a valid movie review.')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')

