import streamlit as st
import joblib
import re


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Vectorizer
# --------------------------------------------------

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💬 Customer Review Sentiment Analysis")

st.write(
    "Analyze customer reviews using Natural Language Processing "
    "and Machine Learning."
)

st.divider()


# --------------------------------------------------
# Model Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "1,000 Reviews")

with col2:
    st.metric("Model", "Naive Bayes")

with col3:
    st.metric("Accuracy", "80.50%")


st.divider()


# --------------------------------------------------
# Review Input
# --------------------------------------------------

st.subheader("📝 Enter Customer Review")

review = st.text_area(
    "Write your review below:",
    placeholder="Example: The product quality is excellent and I am very happy with my purchase.",
    height=160
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():

        st.warning("⚠️ Please enter a customer review.")

    else:

        # Clean review
        cleaned_review = clean_text(review)

        # Convert text into TF-IDF
        review_tfidf = tfidf.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(review_tfidf)[0]

        confidence = max(probabilities) * 100


        st.divider()

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.success("😊 Positive Sentiment")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "The model predicts that this review expresses "
                "a positive opinion."
            )

        else:

            st.error("😞 Negative Sentiment")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "The model predicts that this review expresses "
                "a negative opinion."
            )


# --------------------------------------------------
# Examples
# --------------------------------------------------

st.divider()

st.subheader("💡 Try These Examples")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 😊 Positive")

    st.info(
        "The product is amazing, excellent quality and "
        "worth every penny."
    )


with col2:

    st.markdown("### 😞 Negative")

    st.warning(
        "The product is terrible and completely "
        "waste of money."
    )


# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

st.subheader("🔧 Project Details")

st.write(
    """
    **Technologies Used**
    
    • Python  
    • Pandas  
    • NLTK / Text Processing  
    • TF-IDF Vectorization  
    • Naive Bayes  
    • Scikit-learn  
    • Streamlit
    """
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Review Sentiment Analysis | "
    "Natural Language Processing & Machine Learning"
)