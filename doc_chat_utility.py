import os

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


# Get the current working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the LLM with the Ollama model
llm = Ollama(
    model="llama3.1:8b",
    temperature=0
)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

def get_answer(file_name, query):
    file_path = f"{working_directory}/{file_name}"

    # Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Create text chunks using a character-based splitter
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    
    text_chunks = text_splitter.split_documents(documents)

    # Generate vector embeddings from the text chunks
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    # Create a RetrievalQA chain
    QA_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=knowledge_base.as_retriever()
    )

    # Query the chain and get the response
    response = QA_chain.run(query)

    return response
