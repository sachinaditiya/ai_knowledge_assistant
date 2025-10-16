# -*- coding: utf-8 -*-
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import re

# =============================
# Streamlit App Title
# =============================
st.set_page_config(page_title="🧠 Agentic AI Assistant", layout="wide")
st.title("🧠 Agentic AI Assistant — Multi-PDF + Voice Input + Custom Output")

# =============================
# OpenAI API Key Input
# =============================
st.sidebar.header("🔑 OpenAI API Key")
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if api_key_input:
    st.session_state.openai_api_key = api_key_input
    st.sidebar.success("✅ API key entered successfully.")
else:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# =============================
# Output Type Selection
# =============================
st.sidebar.header("🎯 Output Options")
output_type = st.sidebar.radio("Choose output type:", ["Text Only", "Audio Only", "Text + Audio"])
st.session_state.output_type = output_type

# =============================
# Voice Accent / Language Input
# =============================
st.sidebar.header("🎤 Voice Accent / Language (Optional)")
accent = st.sidebar.selectbox("Choose a voice accent:", ["Default (en)", "US", "UK", "India"])
custom_lang = st.sidebar.text_input("Or type a language code (e.g., en, hi, fr):", "")

def get_lang_code(accent, custom_lang):
    if custom_lang.strip():
        return custom_lang.strip()
    mapping = {
        "Default (en)": "en",
        "US": "en",
        "UK": "en",
        "India": "en"
    }
    return mapping.get(accent, "en")

voice_lang = get_lang_code(accent, custom_lang)

# =============================
# Initialize session state
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# =============================
# PDF Upload Section
# =============================
st.subheader("📄 Upload PDF Files")
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
pdf_text = ""
if uploaded_files:
    with st.spinner("Processing PDF(s)..."):
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text
    st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

# =============================
# Voice Input Controls
# =============================
st.subheader("🎙️ Voice Input (Optional)")
st.caption("🎧 Note: Voice recording works only in local environments. May not work on Streamlit Cloud.")

# Wrap microphone initialization in try-except to avoid crashing in Streamlit Cloud
mic_available = True
try:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
except Exception:
    mic_available = False
    st.info("Voice recording not available in this environment.")

if mic_available:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Recording"):
            st.info("Listening... please speak clearly.")
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            try:
                query_text = recognizer.recognize_google(audio)
                st.session_state.user_question = query_text
                st.success(f"🗣️ Recognized: {query_text}")
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand your voice.")
            except sr.RequestError:
                st.error("Speech recognition service error.")
    with col2:
        if st.button("🛑 Stop Recording"):
            st.info("Recording stopped.")

# =============================
# Question Input
# =============================
st.subheader("💬 Ask a Question")
user_question = st.text_input(
    "Type or edit your question:",
    value=st.session_state.get("user_question", ""),
    placeholder="Ask something about the uploaded PDF..."
)

# =============================
# Function to clean text for embeddings
# =============================
def clean_for_embedding(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F700-\U0001F77F"  
        u"\U0001F780-\U0001F7FF"  
        u"\U0001F800-\U0001F8FF"  
        u"\U0001F900-\U0001F9FF"  
        u"\U0001FA00-\U0001FA6F"  
        u"\U0001FA70-\U0001FAFF"  
        u"\U00002700-\U000027BF"  
        u"\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

# =============================
# Function to convert text to speech
# =============================
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

# =============================
# Get Answer Button
# =============================
if st.button("✨ Get Answer"):
    if not pdf_text:
        st.warning("Please upload at least one PDF first!")
    elif not user_question.strip():
        st.warning("Please enter or speak a question first!")
    else:
        with st.spinner("Thinking..."):
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(pdf_text)
            cleaned_chunks = [clean_for_embedding(chunk) for chunk in chunks]

            try:
                # Attempt FAISS + embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
                vectorstore = FAISS.from_texts(cleaned_chunks, embeddings)
                docs = vectorstore.similarity_search(user_question, k=3)
                context = "\n".join([doc.page_content for doc in docs])

                llm = OpenAI(openai_api_key=st.session_state.openai_api_key, temperature=0)
                answer = llm(
                    f"Answer the question using ONLY the following context:\n{context}\n"
                    f"Question: {user_question}\nAnswer:"
                )

            except Exception as e:
                # Fallback: LLM answer without FAISS context
                st.warning("Could not use FAISS/embeddings. Falling back to direct LLM response.")
                llm = OpenAI(openai_api_key=st.session_state.openai_api_key, temperature=0)
                answer = llm(f"Answer this question:\n{user_question}")

            # Save to chat history
            st.session_state.chat_history.append({
                "user": user_question,
                "bot": answer,
                "output_type": st.session_state.output_type
            })

# =============================
# Display chat history
# =============================
st.subheader("💬 Chat History")
for chat in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {chat['user']}")
    if chat["output_type"] in ["Text Only", "Text + Audio"]:
        st.markdown(f"**Bot:** {chat['bot']}")
    if chat["output_type"] in ["Audio Only", "Text + Audio"]:
        audio_data = text_to_speech(chat['bot'], lang=voice_lang)
        st.audio(audio_data, format="audio/mp3")
    st.markdown("---")

# =============================
# Footer
# =============================
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; font-size: 14px; color: gray;">
        Made with ❤️ by Sachin Aditiya | 
        <a href='https://www.linkedin.com/in/sachin-aditiya-b-7691b314b/' target='_blank'>Connect with me on LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
