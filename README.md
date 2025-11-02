# 🛡️ AIShield — AI Plagiarism Detector

AIShield detects whether a given text or sentence is AI-generated or human-written using NLP and Machine Learning.

## Features
- Custom trained Logistic Regression model with TF-IDF n-grams
- Analyzes text sentence by sentence
- Highlights AI-like sentences in red, human-like in green
- Provides overall AI probability score
- Built with Streamlit for interactive web interface

## Tech Stack
- Python, Scikit-Learn, Streamlit
- NLTK for sentence tokenization
- Joblib for model persistence

## How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Train the model:  
   `python train_model.py`
3. Run the app:  
   `streamlit run app.py`
