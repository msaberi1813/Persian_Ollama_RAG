import streamlit as st
import uuid
from model import LibreChatModel

st.set_page_config(
    page_title="LibreChat AI",
    layout="wide"
)

# ---------- Session State ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "model" not in st.session_state:
    st.session_state.model = LibreChatModel()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧠 LibreChat")

    if st.button("➕ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = []
        st.session_state.current_chat_id = chat_id

    st.divider()
    st.subheader("Chats")

    for cid in reversed(list(st.session_state.chats.keys())):
        messages = st.session_state.chats[cid]
        name = messages[0]["content"][:25] if messages else "New Chat"
        if st.button(name, key=cid, use_container_width=True):
            st.session_state.current_chat_id = cid

# ---------- Main Chat ----------
if not st.session_state.current_chat_id:
    st.info("یک چت جدید بساز یا از سایدبار انتخاب کن.")
    st.stop()

chat_id = st.session_state.current_chat_id
messages = st.session_state.chats[chat_id]

# Displaying previous messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Recieving user input
if user_input := st.chat_input("سوال خود را بپرس…"):
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("در حال فکر کردن..."):
            try:
                # Result by: Few-shot + RAG + memory
                answer = st.session_state.model.get_response(
                    user_input,
                    messages
                )
            except Exception as e:
                answer = f"خطا در پردازش: {e}"

            st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})
