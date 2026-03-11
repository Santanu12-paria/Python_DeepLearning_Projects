import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import pandas as pd
import io

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🌟",
    layout="centered"
)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.write("Analyze sentiment of text or speech using deep learning.")
    st.markdown("---")
    st.write("**Model Details:**")
    st.write("- Max tokens: 200")
    st.write("- Output: Positive / Negative")
    st.markdown("---")
    st.write("**Tips:**")
    st.write("- Speak clearly into mic")
    st.write("- Keep text under 500 chars")
    st.write("- Needs internet for voice")

# ── Load model ─────────────────────────────────────────────
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ── Session state for history ──────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Title ──────────────────────────────────────────────────
st.title("🌟 Text & Audio Sentiment Analysis 🌟")
st.markdown("---")

# ── Helper function ────────────────────────────────────────
def predict_sentiment(text):
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=200)
    prediction = model.predict(padded)
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    confidence = float(prediction[0]) if sentiment == "Positive" else 1 - float(prediction[0])
    return sentiment, confidence

# ── Input mode ─────────────────────────────────────────────
input_mode = st.radio("Choose input method:", ["⌨️ Type Text", "🎙️ Speak"])
st.markdown("")

user_input = ""
analyze_button = False

if input_mode == "⌨️ Type Text":
    user_input = st.text_area("Enter your text:", "", height=150)

    # Word & char count
    col1, col2 = st.columns(2)
    col1.caption(f"📝 Words: {len(user_input.split()) if user_input else 0}")
    col2.caption(f"🔤 Characters: {len(user_input)}")

    if len(user_input) > 500:
        st.warning("⚠️ Long text may reduce accuracy.")

    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

else:
    st.info("🎙️ Click the mic below to record")
    audio_bytes = st.audio_input("Record your message")

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        recognizer = sr.Recognizer()

        with sr.AudioFile(io.BytesIO(audio_bytes.read())) as source:
            audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"📝 Transcribed: **{user_input}**")
                analyze_button = True
            except sr.UnknownValueError:
                st.error("❌ Could not understand. Please speak clearly.")
            except sr.RequestError:
                st.error("❌ Internet needed for speech recognition.")

# ── Results ────────────────────────────────────────────────
if analyze_button and user_input.strip():
    sentiment, confidence = predict_sentiment(user_input)

    st.markdown("---")
    st.markdown("### 📊 Result")

    if sentiment == "Positive":
        emoji, color = "😊", "green"
        if confidence > 0.90:
            st.balloons()
    else:
        emoji, color = "😞", "red"

    col1, col2 = st.columns(2)
    col1.markdown(f"<p style='color:{color}; font-size:26px;'>{emoji} <b>{sentiment}</b></p>", unsafe_allow_html=True)
    col2.metric("Confidence", f"{confidence:.2%}")

    st.progress(confidence)

    # Save to history
    st.session_state.history.append({
        "Text": user_input,
        "Sentiment": sentiment,
        "Confidence": f"{confidence:.2%}"
    })

# ── History ────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Analysis History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download History as CSV", csv, "sentiment_history.csv", "text/csv")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()