# Import library
import streamlit as st
import pickle
import os
from utils import JSONParser, Preprocessor, bot_response

# Load model
model_path = "./model/model_chatbot.pkl"
with open(model_path, "rb") as model_file:
    pipeline = pickle.load(model_file)

# Load JSON data
data_path = "./dataset/data.json"
jp = JSONParser()
jp.parse(data_path)

# Preprocessor instance
pcsr = Preprocessor()

# Streamlit app
st.title("Chatbot Interface")
st.write("Berinteraksi dengan bot kami di bawah ini:")

# Chat input
user_input = st.text_input("Anda >>", placeholder="Ketik pesan di sini...")

# Chat response
if user_input:
    response, tag = bot_response(user_input, pipeline, jp, pcsr)
    st.write(f"**Bot >>** {response}")
    pcsr.save_to_history(user_input, response)

# Save chat history
if st.button("Simpan Riwayat Percakapan"):
    pcsr.save_history_to_file("chat_history.json")
    st.success("Riwayat percakapan telah disimpan ke file `chat_history.json`.")

# Display chat history
if st.checkbox("Tampilkan Riwayat Percakapan"):
    chat_history = pcsr.get_chat_history()
    if chat_history:
        st.json(chat_history)
    else:
        st.write("Belum ada riwayat percakapan.")
