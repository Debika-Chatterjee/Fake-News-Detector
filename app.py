import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# -------------------------------
# Load saved model & vectorizer
# -------------------------------
model = joblib.load("fake_news_logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# Text Preprocessing
# -------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.subheader("Logistic Regression + TF-IDF")
st.write("Enter a news article below to check whether it is **Fake or Real**.")

news_input = st.text_area("📝 Enter News Text", height=200)

if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        processed_text = preprocess(news_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

st.markdown("---")
st.markdown("📌 **Model:** Logistic Regression")
st.markdown("📌 **Feature Extraction:** TF-IDF")
st.markdown("📌 **Accuracy:** ~95% (Validation)")
