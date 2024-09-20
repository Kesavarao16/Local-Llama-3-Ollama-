import os
import streamlit as st
from doc_chat_utility import get_answer

# Get the working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Document Q&A - LLAMA 3 - OLLAMA",
    page_icon="🗎",
    layout='centered'
)

st.title("Document Q&A - LLAMA 3 - OLLAMA")

# File uploader
uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

# Input for user query
user_query = st.text_input("Ask your question")

# Only proceed when the 'Run' button is clicked
if st.button("Run"):
    if uploaded_file is not None:
        if user_query.strip() == "":
            st.error("Please enter a question.")
        else:
            # Read the file data
            bytes_data = uploaded_file.read()
            file_name = uploaded_file.name
            
            # Save the uploaded file to the working directory
            file_path = os.path.join(working_directory, file_name)
            with open(file_path, 'wb') as f:
                f.write(bytes_data)

            # Get the answer using the utility function
            try:
                answer = get_answer(file_name, user_query)
                st.success(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please upload a file before running the query.")
