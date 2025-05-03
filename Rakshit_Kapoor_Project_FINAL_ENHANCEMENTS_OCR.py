# Analysis of Cancer-Causing Ingredients in Food Products (Steps A to G)
# Designed to run in Google Colab
# Author: Adapted from Rakshit_Kapoor_Project_FINAL_OCR.py

# Step A: System Setup and Library Configuration
# Install required libraries
!pip install pytesseract opencv-python-headless pyzbar numpy pandas scikit-learn nltk plotly playsound

# Import libraries
import os
import cv2
import pytesseract
from pyzbar import pyzbar
import numpy as np
import pandas as pd
import sqlite3
import requests
import nltk
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from playsound import playsound

# Configure Pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Default for Colab
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("System setup and library configuration complete.")

# Step B: Data Source Integration
# Create a sample SQLite database in-memory for demo
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE carcinogens (
        ingredient TEXT,
        risk_percentage REAL
    )
""")
sample_carcinogens = [
    ('red 40', 80.0),
    ('aspartame', 70.0),
    ('natural flavor', 10.0)
]
cursor.executemany("INSERT INTO carcinogens (ingredient, risk_percentage) VALUES (?, ?)", sample_carcinogens)
conn.commit()
carcinogen_df = pd.read_sql_query("SELECT * FROM carcinogens", conn)
logging.info("Connected to in-memory SQLite database. Carcinogen data loaded:\n%s", carcinogen_df)

# Simulated UPC API lookup
def lookup_upc_data(upc_code):
    try:
        # Mock API response for demo
        logging.info("Simulated UPC lookup for code: %s", upc_code)
        return {'ingredients': 'red 40, sugar, aspartame'}
    except Exception as e:
        logging.exception("Error during UPC lookup: %s", e)
        return None

logging.info("Data Source Integration complete.")

# Step C: Image and Text Data Extraction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Failed to load image: %s", image_path)
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image_path, lang='eng'):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return ""
    text = pytesseract.image_to_string(processed_image, lang=lang)
    logging.info("Extracted text from image.")
    return text

def extract_barcode_data(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Failed to load image for barcode: %s", image_path)
        return []
    barcodes = pyzbar.decode(image)
    barcode_data = [barcode.data.decode('utf-8') for barcode in barcodes]
    logging.info("Extracted barcode data: %s", barcode_data)
    return barcode_data

# Step D: Text Preprocessing and Standardization
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize

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

# Step E: Ingredient Matching and Risk Classification
def match_ingredients(cleaned_text, carcinogen_df):
    matched = []
    for index, row in carcinogen_df.iterrows():
        if row['ingredient'].lower() in cleaned_text:
            matched.append({'ingredient': row['ingredient'], 'risk_percentage': row['risk_percentage']})
    logging.info("Matching complete. Found %d matched ingredients.", len(matched))
    return matched

def classify_risk(features, model, vectorizer):
    try:
        X = vectorizer.transform([features])
        risk_score = model.predict_proba(X)[0][1] * 100
        logging.info("Risk classification complete. Score: %.2f%%", risk_score)
        return risk_score
    except Exception as e:
        logging.exception("Error in risk classification: %s", e)
        return None

# Train a sample Random Forest model
sample_data = ["red 40 aspartame", "natural flavor sugar", "aspartame syrup"]
sample_labels = [0.8, 0.1, 0.7]
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sample_data)
y_train = [1 if x > 0.5 else 0 for x in sample_labels]
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
logging.info("ML model trained for risk classification.")

# Step F: Real-Time Alert Generation and User Interaction
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

def play_audio_alert(alert_message):
    print("AUDIO ALERT:", alert_message)

def log_health_journal(product_name, risk_score, decision):
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

# Step G: Data Visualization and Reporting
def visualize_risk_data(carcinogen_df):
    fig = px.bar(carcinogen_df, x="ingredient", y="risk_percentage",
                 title="Carcinogen Risk Percentages for Ingredients",
                 labels={"ingredient": "Ingredient", "risk_percentage": "Risk (%)"})
    fig.show()

# Example execution
if __name__ == "__main__":
    # Upload a sample image in Colab and specify path
    sample_image = 'sample_label.jpg'  # Replace with actual path after uploading
    label_text = extract_text_from_image(sample_image)
    cleaned_text = preprocess_ingredient_text(label_text)
    barcode_data = extract_barcode_data(sample_image)
    matched_ingredients = match_ingredients(cleaned_text, carcinogen_df)
    risk_score = classify_risk(cleaned_text, model, vectorizer)
    alert_message = generate_alert(risk_score)
    play_audio_alert(alert_message)
    log_health_journal("Sample Product", risk_score, "DO NOT CONSUME")
    visualize_risk_data(carcinogen_df)
    logging.info("Steps A to G executed successfully.")