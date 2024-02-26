import os
import sys
try:
    sys.path.append("gemma_pytorch")
except:
    raise ValueError("There is no gemma library")

from glob import glob
from datetime import datetime
from collections import deque
import joblib

import streamlit as st

import torch

from gemma_pytorch.gemma.config import get_config_for_2b,get_config_for_7b
from gemma_pytorch.gemma.model import GemmaForCausalLM

VARIANT = "2b-it"
MACHINE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = torch.device(MACHINE_TYPE)
DATA_PATH = "data"
OUTPUT_LEN = 100

@st.cache_resource
def load_model(tokenizer_path, ckpt_path):
    model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    model_config.tokenizer = tokenizer_path
    model_config.quant = "quant" in VARIANT
    
    torch.set_default_dtype(model_config.get_dtype())
    
    model = GemmaForCausalLM(model_config)
    model.load_weights(ckpt_path)
    model = model.to(DEVICE).eval()
    
    return model


def make_new_chat(chat_name):
    st.session_state.chat_id = chat_name
    # try
    st.session_state.chat_history[chat_name] = []


def main():
    # setting
    
    st.set_page_config(
        page_title="Gemma Chat Bot",
        page_icon=":books:"
    )
    
    with st.spinner("Load Model..."):
        weight_dir = f"gemma-{VARIANT}"
        tokenizer_path = os.path.join(weight_dir, "tokenizer.model")
        ckpt_path = os.path.join(weight_dir, f"gemma-{VARIANT}.ckpt")
        model = load_model(tokenizer_path, ckpt_path)
    
    st.title("Gemma ChatBot")
    
    
    
    # ---session state init---
    
    if "chat_history" not in st.session_state:
        chat_paths = glob(f"{DATA_PATH}/*")
        st.session_state.chat_history = {}
        for path in chat_paths:
            chat_list = joblib.load(path)
            chat_name = os.path.basename(path)
            st.session_state.chat_history[chat_name] = chat_list
            
    
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = f"New-Chat-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        st.session_state.chat_history[st.session_state.chat_id] = []
    
        
    # ---

    with st.sidebar:
        st.write("## Chats History")
        st.session_state.chat_id = st.selectbox(
            label="pick a chat",
            index=len(st.session_state.chat_history)-1,
            options=list(st.session_state.chat_history.keys()),
            # format_func=
            placeholder="_"
        )
        
        chat_name = st.text_input(
            label="Input Chat Name"
        )
        # 버튼 두 번 클릭해야 스크립트가 재실행 되는 현상 해결
        st.button("Make New Chat", on_click=make_new_chat, args=[chat_name])

        save_chat = st.button("Save")
    
    st.subheader(st.session_state.chat_id)
    
    messages = st.session_state.chat_history.get(st.session_state.chat_id, None)
    
    if save_chat:
        joblib.dump(messages, f"{DATA_PATH}/{st.session_state.chat_id}")
        st.info("저장되었습니다")
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("say something"):
        messages.append({
            "role": "user",
            "content": query
        })
        
        with st.chat_message("user"):
            st.markdown(query)
        
        chat_input = [f"{data['role']}: {data['content']}" for data in messages]
        chat_input = "\n".join(chat_input)
        with st.chat_message("chatbot"):
            with st.spinner("Thinking..."):
                result = model.generate(
                    chat_input,
                    device=DEVICE,
                    output_len=OUTPUT_LEN
                )
                st.markdown(result)
        
        messages.append({"role": "chatbot", "content": result})

if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    main()