# Food Carcinogen Risk Analysis Web Application (Step H)
# Designed to run in VSCode with Streamlit
# Author: Adapted from Rakshit_Kapoor_Project_FINAL_OCR.py

import streamlit as st
import cv2
import numpy as np
import pytesseract
from pyzbar import pyzbar
import pandas as pd
import sqlite3
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Streamlit application setup initiated.")

# Initialize NLTK
nltk.download('punkt')

# Load pre-trained model and vectorizer (simulated from Step E)
sample_data = ["red 40 aspartame", "natural flavor sugar", "aspartame syrup"]
sample_labels = [0.8, 0.1, 0.7]
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sample_data)
y_train = [1 if x > 0.5 else 0 for x in sample_labels]
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Connect to SQLite database
db_path = 'carcinogen_risk_database.sqlite3'
conn = sqlite3.connect(db_path)
carcinogen_df = pd.read_sql_query("SELECT * FROM carcinogens", conn)
logging.info("Connected to SQLite database.")

# Image preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Text extraction
def extract_text_from_image(image, lang='eng'):
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, lang=lang)
    logging.info("Extracted text from image.")
    return text

# Barcode extraction
def extract_barcode_data(image):
    barcodes = pyzbar.decode(image)
    barcode_data = [barcode.data.decode('utf-8') for barcode in barcodes]
    logging.info("Extracted barcode data: %s", barcode_data)
    return barcode_data

# Text preprocessing
def preprocess_ingredient_text(raw_text):
    try:
        text = raw_text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        processed_text = " ".join(tokens)
        logging.info("Ingredient text preprocessed.")
        return processed_text
    except Exception as e:
        logging.exception("Error during text preprocessing: %s", e)
        return ""

# Risk classification
def classify_risk(features, model, vectorizer):
    try:
        X = vectorizer.transform([features])
        risk_score = model.predict_proba(X)[0][1] * 100
        logging.info("Risk classification complete. Score: %.2f%%", risk_score)
        return risk_score
    except Exception as e:
        logging.exception("Error in risk classification: %s", e)
        return None

# Alert generation
def generate_alert(risk_score):
    if risk_score is None:
        alert_message = "Risk score could not be determined."
    elif risk_score > 70:
        alert_message = f"Warning: High Cancer Risk ({risk_score:.2f}%) detected! Consider avoiding this product."
    elif risk_score > 40:
        alert_message = f"Alert: Moderate Cancer Risk ({risk_score:.2f}%). Please review product details."
    else:
        alert_message = f"Risk is low ({risk_score:.2f}%). Product seems safe to consume in moderation."
    logging.info("Alert generated: %s", alert_message)
    return alert_message

# Health journal logging
def log_health_journal(product_name, risk_score, decision):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            risk_score REAL,
            decision TEXT,
            scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("INSERT INTO health_journal (product_name, risk_score, decision) VALUES (?, ?, ?)",
                   (product_name, risk_score, decision))
    conn.commit()
    logging.info("Health journal updated for %s with decision: %s", product_name, decision)

# Streamlit app
def run_streamlit_app():
    st.title("Food Carcinogen Risk Analysis")
    st.write("Upload a food label or barcode image to analyze carcinogenic ingredients.")

    # File uploader
    uploaded_file = st.file_uploader("Upload Food Label/Barcode Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR", caption="Uploaded Image")

        # Extract and process text
        extracted_text = extract_text_from_image(image)
        cleaned_text = preprocess_ingredient_text(extracted_text)
        st.write("Extracted & Cleaned Text:")
        st.text(cleaned_text)

        # Extract barcode
        barcode_data = extract_barcode_data(image)
        st.write("Barcode Data:", barcode_data)

        # Classify risk and generate alert
        risk_score = classify_risk(cleaned_text, model, vectorizer)
        alert_message = generate_alert(risk_score)
        st.write(f"Risk Score: {risk_score:.2f}%")
        st.write("Alert Message:", alert_message)
        st.success("Push Notification: " + alert_message)

        # Health journal
        decision = st.radio("Your Decision:", ["DO NOT CONSUME", "CONSUME ANYWAY"])
        if st.button("Log Scan"):
            log_health_journal("Uploaded Product", risk_score, decision)
            st.info("Health journal updated.")

if __name__ == "__main__":
    run_streamlit_app()