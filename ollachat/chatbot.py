import os
import json
import datetime
import time

import streamlit as st
import ollama

try:
    OLLAMA_MODELS = ollama.list()["models"]
except Exception as e:
    st.warning("Please make sure Ollama is installed first. See https://ollama.ai for more details.")
    st.stop()

def st_ollama(model_name, user_question, chat_history_key):
    # Add custom CSS for clean styling with buffer
    st.markdown("""
        <style>
        .thinking-expander {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(0, 0, 0, 0.02);
        }
        .streamlit-expanderHeader {
            border: none !important;
            background-color: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if chat_history_key not in st.session_state.keys():
        st.session_state[chat_history_key] = []

    print_chat_history_timeline(chat_history_key)
    
    if user_question:
        st.session_state[chat_history_key].append({"content": f"{user_question}", "role": "user"})
        with st.chat_message("question", avatar="🧑‍🚀"):
            st.write(user_question)

        messages = [dict(content=message["content"], role=message["role"]) for message in st.session_state[chat_history_key]]

        # Streaming response
        with st.chat_message("response", avatar="🤖"):
            chat_box = st.empty()
            thinking_box = st.empty()
            response_box = st.empty()
            
            def process_stream(stream):
                nonlocal thinking_buffer, response_buffer, in_thinking_mode
                
                for chunk in stream:
                    if "<think>" in chunk and not in_thinking_mode:
                        in_thinking_mode = True
                        thinking_buffer = chunk.replace("<think>", "")
                        
                        # Introduce buffer before displaying reasoning
                        with thinking_box.container():
                            with st.expander("🧠 Reasoning Process", expanded=False):
                                st.markdown('<div class="thinking-expander">', unsafe_allow_html=True)
                                st.write("🤔 Thinking...")  # Placeholder text
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        time.sleep(1.5)  # Buffer delay
                        
                        with thinking_box.container():
                            with st.expander("🧠 Reasoning Process", expanded=False):
                                st.markdown('<div class="thinking-expander">', unsafe_allow_html=True)
                                st.write(thinking_buffer)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    elif "</think>" in chunk and in_thinking_mode:
                        in_thinking_mode = False
                        thinking_buffer += chunk.replace("</think>", "")

                        with thinking_box.container():
                            with st.expander("🧠 Reasoning Process", expanded=False):
                                st.markdown('<div class="thinking-expander">', unsafe_allow_html=True)
                                st.write(thinking_buffer)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    elif in_thinking_mode:
                        thinking_buffer += chunk
                        with thinking_box.container():
                            with st.expander("🧠 Reasoning Process", expanded=False):
                                st.markdown('<div class="thinking-expander">', unsafe_allow_html=True)
                                st.write(thinking_buffer)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        response_buffer += chunk
                        response_box.write(response_buffer)
            
            # Initialize buffers
            thinking_buffer = ""
            response_buffer = ""
            in_thinking_mode = False
            
            def llm_stream(response):
                response = ollama.chat(model_name, messages, stream=True)
                for chunk in response:
                    yield chunk['message']['content']
            
            # Process the stream
            process_stream(llm_stream(messages))

            final_content = f"<think>{thinking_buffer}</think>\n\n{response_buffer}"
            st.session_state[chat_history_key].append({"content": final_content, "role": "assistant"})
            return final_content

def print_chat_history_timeline(chat_history_key):
    for message in st.session_state[chat_history_key]:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="🧑‍🚀"): 
                st.markdown(content, unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                if "<think>" in content and "</think>" in content:
                    thinking, response = content.split("</think>", 1)
                    thinking = thinking.replace("<think>", "").strip()
                    response = response.strip()
                    
                    with st.expander("🧠 Reasoning Process", expanded=False):
                        st.markdown('<div class="thinking-expander">', unsafe_allow_html=True)
                        st.write(thinking)
                        st.markdown('</div>', unsafe_allow_html=True)
                    if response:
                        st.write(response)
                else:
                    st.markdown(content, unsafe_allow_html=True)

def assert_models_installed():
    if len(OLLAMA_MODELS) < 1:
        st.sidebar.warning("No models found. Please install at least one model e.g. `ollama run llama2`")
        st.stop()

def select_model():
    model_names = [model["name"] for model in OLLAMA_MODELS]
    llm_name = st.sidebar.selectbox(f"Choose Agent (available {len(model_names)})", [""] + model_names)
    return llm_name

def save_conversation(llm_name, conversation_key):
    OUTPUT_DIR = "llm_conversations"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if st.session_state[conversation_key]:
        if st.sidebar.button("Save conversation"):
            filename = f"{OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{llm_name.replace(':', '-')}"
            with open(f"{filename}.json", "w") as f:
                json.dump(st.session_state[conversation_key], f, indent=4)
            st.success(f"Conversation saved to {filename}.json")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Ollama Chat", page_icon="🦙")
    st.sidebar.title("Ollama Chat 🦙")
    llm_name = select_model()
    assert_models_installed()
    if not llm_name: st.stop()
    conversation_key = f"model_{llm_name}"
    prompt = st.chat_input(f"Ask '{llm_name}' a question ...")
    st_ollama(llm_name, prompt, conversation_key)
    save_conversation(llm_name, conversation_key)
