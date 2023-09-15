import streamlit as st
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import time
import threading

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("Dr. Victor Baugaredde (Stroke Specialist)")

# Initialize Langchain components
loader = TextLoader("stroke.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=5)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if Chroma database already exists, if not, create it
if not st.session_state.get("chroma_db"):
    # Create the Chroma database with the persist_directory option
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
    st.session_state["chroma_db"] = db
else:
    # Load the Chroma database from session state
    db = st.session_state["chroma_db"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to auto-save the Chroma database at regular intervals
def auto_save_chroma_db():
    while True:
        time.sleep(600)  # Save every 10 minutes (adjust as needed)
        db.save()

# Start the auto-save thread
if "auto_save_thread" not in st.session_state:
    st.session_state.auto_save_thread = threading.Thread(target=auto_save_chroma_db)  # Use threading.Thread

    # Set the thread as a daemon so it doesn't block program exit
    st.session_state.auto_save_thread.daemon = True

    # Start the thread
    st.session_state.auto_save_thread.start()

# Accept user input
user_input = st.chat_input("You:")
if user_input:
    user_input = str(user_input)  # Ensure user_input is a string

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(f"You: {user_input}")

    # Check if the user's input contains "thanks" or "thank you"
    if "thanks" in user_input.lower() or "thank you" in user_input.lower():
        # Respond with a specific message
        assistant_response = "You're welcome! Is there anything else you want to ask?"
    else:
        # Generate assistant response based on user input
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Generate the assistant's response using the loaded Chroma database
            response = db.similarity_search(user_input)
            if response:
                assistant_response = response[0].page_content
            else:
                assistant_response = "I couldn't find information related to your query."

            # Simulate typing with a cursor
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(f"Victor: {full_response}▌")
            message_placeholder.markdown(f"Victor: {full_response}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
