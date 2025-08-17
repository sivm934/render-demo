import streamlit as st
import pickle
import string
import os
import shutil
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# NLTK setup for Render with punkt_tab workaround
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required corpora
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# Create dummy punkt_tab if needed
try:
    nltk.data.find('tokenizers/punkt_tab/english/pickle')
except LookupError:
    src = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    dst = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab')
    shutil.copytree(src, dst, dirs_exist_ok=True)  # overwrite if exists

# -----------------------------

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
