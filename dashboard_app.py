import os
print(os.listdir())
import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load models & data
# -----------------------------
@st.cache_resource
def load_models():
    category_model = joblib.load("email_classifier_model.pkl")
    category_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    urgency_model = joblib.load("urgency_model.pkl")
    urgency_vectorizer = joblib.load("urgency_vectorizer.pkl")
    return category_model, category_vectorizer, urgency_model, urgency_vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("final_email_dataset_expanded.csv")

category_model, category_vectorizer, urgency_model, urgency_vectorizer = load_models()
df = load_data()

# -----------------------------
# UI - Title
# -----------------------------
st.title("📧 AI Smart Email Classifier")
st.markdown("### 🚀 Powered by Machine Learning (SVM + NLP)")

st.divider()

# -----------------------------
# 📊 Dataset Insights
# -----------------------------
st.header("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Category Distribution")
    st.bar_chart(df["category"].value_counts())

with col2:
    st.subheader("⚡ Urgency Distribution")
    st.bar_chart(df["urgency"].value_counts())

st.divider()

# -----------------------------
# ✍️ Input Section
# -----------------------------
st.header("✍️ Enter Email")

# Sample button
if st.button("Test Sample Email"):
    st.session_state.email_text = "My payment failed please fix immediately"

email_text = st.text_area(
    "Email Text",
    value=st.session_state.get("email_text", "")
)

# -----------------------------
# 🔍 Prediction
# -----------------------------
if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:

        # -----------------------------
        # Predictions
        # -----------------------------
        cat_vec = category_vectorizer.transform([email_text])
        category = category_model.predict(cat_vec)[0]

        urg_vec = urgency_vectorizer.transform([email_text])
        urgency = urgency_model.predict(urg_vec)[0]

        st.divider()

        # -----------------------------
        # 🎯 Results
        # -----------------------------
        st.subheader("🔍 AI Analysis Result")

        st.markdown(f"### 📂 Category: **{category.upper()}**")
        st.markdown(f"### ⚡ Urgency: **{urgency.upper()}**")

        

        # -----------------------------
        # 🧠 Confidence Score
        # -----------------------------
        try:
            confidence = max(category_model.decision_function(cat_vec)[0])
            st.markdown(f"### 🧠 Confidence Score: **{round(confidence, 2)}**")
        except:
            st.write("Confidence not available")

        # -----------------------------
        # 🔑 Top Keywords
        # -----------------------------
        feature_names = category_vectorizer.get_feature_names_out()
        vector_array = cat_vec.toarray()[0]

        top_indices = vector_array.argsort()[-5:][::-1]

        st.subheader("🔑 Top Keywords Detected")

        for i in top_indices:
            if vector_array[i] > 0:
                st.write(f"👉 {feature_names[i]}")

        # -----------------------------
        # 🎨 Urgency Color Indicator
        # -----------------------------
        if urgency == "high":
            st.error("🚨 High Priority Email")
        elif urgency == "medium":
            st.warning("⚠️ Medium Priority Email")
        else:
            st.success("✅ Low Priority Email")
