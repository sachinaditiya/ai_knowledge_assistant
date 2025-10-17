# -*- coding: utf-8 -*-
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from gtts import gTTS
from io import BytesIO
import re

# =============================
# Streamlit App Config
# =============================
st.set_page_config(page_title="🧠 Agentic AI Assistant", layout="wide")
st.title("🧠 Agentic AI Assistant — Multi-PDF + Voice Input + Custom Output")

# =============================
# API Key Section
# =============================
st.sidebar.header("🔑 OpenAI API Key")
api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not api_key_input:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()
st.session_state.openai_api_key = api_key_input
st.sidebar.success("✅ API key entered successfully.")

# =============================
# Output Options
# =============================
st.sidebar.header("🎯 Output Options")
output_type = st.sidebar.radio("Choose output type:", ["Text Only", "Audio Only", "Text + Audio"])
st.session_state.output_type = output_type

# =============================
# Voice Accent / Language
# =============================
st.sidebar.header("🎤 Voice Accent / Language (Optional)")
accent = st.sidebar.selectbox("Choose a voice accent (optional):", ["Default (en)", "US", "UK", "India"])
custom_lang = st.sidebar.text_input("Or type a language code (e.g., en, hi, fr):", "")

def get_lang_code(accent, custom_lang):
    if custom_lang.strip():
        return custom_lang.strip()
    mapping = {"Default (en)": "en", "US": "en", "UK": "en", "India": "en"}
    return mapping.get(accent, "en")

voice_lang = get_lang_code(accent, custom_lang)

# =============================
# Session State Setup
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =============================
# PDF Upload
# =============================
st.subheader("📄 Upload PDF Files")
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

def clean_for_embedding(text):
    """Remove emojis and unwanted characters."""
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

@st.cache_data
def process_pdfs(files):
    """Extract and clean text from uploaded PDFs."""
    text = ""
    for uploaded_file in files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += clean_for_embedding(page_text)
    return text

if uploaded_files:
    with st.spinner("Processing PDF(s)..."):
        pdf_text = process_pdfs(uploaded_files)
        st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")
else:
    pdf_text = ""

# =============================
# Question Input
# =============================
st.subheader("💬 Ask a Question")
user_question = st.text_input("Ask something about the uploaded PDF...")

# =============================
# Text-to-Speech Function
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
        st.stop()
    if not user_question.strip():
        st.warning("Please enter a question first!")
        st.stop()

    with st.spinner("Processing your question..."):
        if st.session_state.vectorstore is None:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(pdf_text)
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)

        vectorstore = st.session_state.vectorstore
        docs = vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        llm = OpenAI(openai_api_key=st.session_state.openai_api_key, temperature=0)
        prompt = f"""
        You are a knowledgeable assistant. Use only the context below to answer.
        If you don't find the answer in the context, say "The information is not available in the provided PDF(s)."

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """
        answer = llm(prompt)

        st.session_state.chat_history.append({
            "user": user_question,
            "bot": answer,
            "output_type": st.session_state.output_type
        })

# =============================
# Display Chat History
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
