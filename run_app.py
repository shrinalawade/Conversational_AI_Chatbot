import streamlit as st
from predictions import get_prediction

st.set_page_config(page_title="Chat with the Streamlit", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Conversation AI- Medical Diagnosis")
st.button("Clear Chat History", type="primary")

if prompt := st.chat_input("Your question"):
    prompt_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(prompt_message)

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Enter the patients health summary"}]

    else:
        question = st.session_state.messages[len(st.session_state.messages)-1]['content']
        res = {"Predicted Diagnosis": get_prediction(question)}
        message = {"role": "assistant", "content": res}
        st.session_state.messages.append(message)

elif st.button:
    st.session_state.messages = [
        {"role": "assistant", "content": "Enter the patients health summary"}]


for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
