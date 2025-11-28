import streamlit as st
import pickle
import time
import numpy as np
from preprocess import clean_text

# ------------------- Load Saved Model -------------------
# (Ensure your paths are correct)
try:
    vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
    model = pickle.load(open("model/svm_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found! Please check the 'model/' directory.")
    st.stop()

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Map labels to emojis and descriptions for better UI
label_meta = {
    "toxic": {"emoji": "⚠️", "desc": "General Toxicity"},
    "severe_toxic": {"emoji": "☣️", "desc": "Highly Offensive"},
    "obscene": {"emoji": "🔞", "desc": "Sexual/Vulgar"},
    "threat": {"emoji": "💀", "desc": "Violence Threat"},
    "insult": {"emoji": "🤬", "desc": "Personal Attack"},
    "identity_hate": {"emoji": "👺", "desc": "Racism/Discrimination"}
}

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="SafeSpeak AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Background: smooth, lively gradient */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0f1724 0%, #081226 20%, #0b1020 40%, #071427 70%),
                    linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.06) 40%, rgba(236,72,153,0.03) 100%);
        background-attachment: fixed;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #0b1020 !important;
        border: 1px solid #333 !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
    }
    .stTextArea textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.2);
    }

    /* Button Styling */
    /* Button appearance + centered layout (not full width) */
    .stButton { display: flex; justify-content: center; }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
        color: white;
        border: none;
        padding: 12px 36px;
        border-radius: 50px;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
        width: auto !important; /* don't force full column width */
        min-width: 220px;
        display: inline-block;
        margin: 0 auto;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }

    /* Glassmorphism main panel */
    .stContainer {
        max-width: 980px;
        margin: 26px auto !important;
        padding: 28px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 8px 40px rgba(2,6,23,0.6);
        backdrop-filter: blur(8px) saturate(1.1);
        -webkit-backdrop-filter: blur(8px) saturate(1.1);
        color: #e6eef6;
    }

    /* Heading & description inside the panel */
    .stContainer h1, .stContainer p {
        margin: 4px 0 12px 0 !important;
        text-align: center;
    }

    /* Make the card area slightly elevated on desktop */
    @media (min-width: 800px) {
        .stContainer { padding: 40px !important; }
    }

    /* Result Cards */
    .toxic-card {
        background-color: linear-gradient(135deg, rgba(255,75,75,0.08), rgba(255,75,75,0.02));
        border-left: 5px solid #FF4B4B;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #FFCDD2;
        font-weight: 500;
        animation: slideIn 0.5s ease;
    }

    .clean-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
        border: 1px solid rgba(255,255,255,0.06);
        border: 1px solid #4CAF50;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: #C8E6C9;
        animation: fadeIn 0.8s ease;
    }

    @keyframes slideIn {
        from {transform: translateX(-20px); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
with st.sidebar:
    st.image("images\logo3.png", width=200)
    st.markdown("<h3 >SafeSpeak AI</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("This tool uses **Natural Language Processing (NLP)** and **Machine Learning** to detect harmful content in text.")
    
    st.info("**Model Used:** TF-IDF + LinearSVC")
    st.warning("⚠️ **Disclaimer:** AI can make mistakes. Always review flagged content manually.")
    st.markdown("---")
    st.caption("v1.0.0 • Devansh Rahatal")

# ------------------- Main Content -------------------
c1, c2, c3 = st.columns([1, 6, 1])

with c2:
    st.image("images/logo.jpg", width=700)
st.markdown("<h2 style='text-align: center; color: #fff; font: Source Serif';>Toxic Comment Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa; margin-bottom: 30px; font-weight: bold;'>Enter a comment below to scan for toxicity, hate speech, and threats.</p>", unsafe_allow_html=True)

# Container for input
with st.container():
    
    # Text Input
    user_input = st.text_area("Analyze Text", height=150, placeholder="Paste user comment here...", label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("🔍 Scan for Toxicity")
    
    # cleaned up stray HTML wrapper — Streamlit handles the layout for us

# ------------------- Logic & Results -------------------
if analyze_btn:
    if user_input.strip() == "":
        st.toast("⚠️ Please enter some text first!", icon="⚠️")
    else:
        with st.spinner("🤖 AI is analyzing sentiment..."):
            time.sleep(0.8) # Artificial delay for effect
            
            # Preprocess & Predict
            cleaned_text = clean_text(user_input)
            X = vectorizer.transform([cleaned_text])
            preds = model.predict(X)[0]

            # Results Display
            st.markdown("### Analysis Results")
            
            # Check if any toxic label is active
            active_labels = [label for i, label in enumerate(labels) if preds[i] == 1]

            if active_labels:
                # If Toxic
                st.error("🚨 Content Flagged as Toxic")
                
                # Display Grid of Toxic Cards
                cols = st.columns(2)
                for idx, label in enumerate(active_labels):
                    meta = label_meta.get(label, {"emoji": "⚠️", "desc": "Unknown"})
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div class='toxic-card'>
                            <span style='font-size: 20px;'>{meta['emoji']}</span> 
                            <b>{label.replace('_', ' ').title()}</b>
                            <br><span style='font-size: 12px; opacity: 0.8;'>{meta['desc']}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # If Clean
                st.markdown("""
                <div class='clean-card'>
                    <h2 style='margin:0;'>🌿 Clean Content</h2>
                    <p style='margin:5px 0 0 0;'>No toxicity detected. This comment appears safe.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()